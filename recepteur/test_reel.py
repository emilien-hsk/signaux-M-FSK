"""Tests acoustiques réels du modem M-FSK.

PC émetteur (haut-parleur) : joue un texte du corpus
  python test_reel.py emettre --M 4 --texte T1

PC récepteur (micro) : écoute sans savoir quel texte arrive
  python test_reel.py ecouter --M 4 --distance 1 --bruit calme
  -> détecte le message, l'écoute jusqu'au SIGNAL DE FIN (chirp descendant),
     reconnaît le texte du corpus, note le résultat, puis s'arrête.
  Ajouter --continu pour rester à l'écoute de plusieurs messages (Ctrl+C pour arrêter).

Un seul PC + un baffle (joue et enregistre en même temps) :
  python test_reel.py boucle --M 4 --texte T1 --distance 1 --bruit calme

Réanalyser un enregistrement sauvegardé (--texte facultatif) :
  python test_reel.py analyser --M 4 --distance 1 --bruit calme --fichier enregistrements/xxx.wav
"""
import argparse
import csv
import os
import queue
import sys
from collections import deque
from datetime import datetime

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, sosfilt_zi

from mfsk import (FS, TS, emettre, trouver_debut_robuste, detecter_symboles,
                  symboles_vers_bits, bits_vers_texte, chercher_fin)

CORPUS = {
    "T1": "Bonjour",
    "T2": "Portez ce vieux whisky au juge blond qui fume",
    "T3": "Code 4021 : 75% ok ! (test #3)",
    "T4": "Élève, été, ça va ? Où ça ?",
    "T5": "Ce message de test sert à mesurer la robustesse du modem M-FSK sur une phrase plus longue.",
}
# longueur en octets (UTF-8) -> identifiant : les 5 textes ont des longueurs différentes
PAR_LONGUEUR = {len(t.encode("utf-8")): k for k, t in CORPUS.items()}
FICHIER_CSV = "resultats_reels.csv"


# ======================================================================
#  Analyse d'un enregistrement (texte attendu connu OU reconnu tout seul)
# ======================================================================
def identifier(enreg, M, debut):
    """Lit l'octet de longueur et en déduit quel texte du corpus a été envoyé.
    Renvoie (identifiant ou None, texte décodé selon la longueur lue)."""
    k = int(np.log2(M))
    entete = detecter_symboles(enreg, M, debut, int(np.ceil(8 / k)))
    L = int(np.packbits(symboles_vers_bits(entete, M)[:8])[0])
    sym = detecter_symboles(enreg, M, debut, int(np.ceil(8 * (1 + L) / k)))
    recu = bits_vers_texte(symboles_vers_bits(sym, M))
    if L in PAR_LONGUEUR:
        return PAR_LONGUEUR[L], recu
    # octet de longueur abîmé : on compare les bits reçus à ceux de chaque texte du corpus
    nb_entete = int(np.ceil(8 / k))
    taux = {}
    for c, t in CORPUS.items():
        _, sym_c = emettre(t, M)
        sym_r = detecter_symboles(enreg, M, debut, len(sym_c))
        if len(sym_r) < len(sym_c):
            continue
        b_c = symboles_vers_bits(sym_c[nb_entete:], M)
        b_r = symboles_vers_bits(sym_r[nb_entete:], M)
        taux[c] = float(np.mean(b_c != b_r))
    if taux:
        meilleur = min(taux, key=taux.get)
        if taux[meilleur] < 0.2:   # au hasard on aurait ≈ 50 % de bits faux
            return meilleur, recu
    return None, recu  # texte libre, hors corpus


def analyser(enreg, M, texte_id=None):
    """Décode un enregistrement. Renvoie (texte_id, reçu, TEB, TEC) ;
    TEB et TEC valent None si le texte envoyé n'a pas pu être reconnu."""
    debut, methode = trouver_debut_robuste(enreg, M)
    print(f"Synchronisation : {methode} (début du message à {debut / FS:.2f} s)")
    if texte_id is None:
        texte_id, recu = identifier(enreg, M, debut)
        if texte_id is None:
            return None, recu, None, None
    texte = CORPUS[texte_id]
    _, sym_envoyes = emettre(texte, M)
    sym_recus = detecter_symboles(enreg, M, debut, len(sym_envoyes))
    if len(sym_recus) < len(sym_envoyes):  # enregistrement coupé trop tôt
        sym_recus = np.concatenate([sym_recus, np.zeros(len(sym_envoyes) - len(sym_recus), dtype=int)])
    b_tx = symboles_vers_bits(sym_envoyes, M)
    b_rx = symboles_vers_bits(sym_recus, M)
    recu = bits_vers_texte(b_rx)
    teb = float(np.mean(b_tx != b_rx))
    n = max(len(texte), len(recu))
    tec = sum(1 for i in range(n) if i >= len(recu) or i >= len(texte) or recu[i] != texte[i]) / n
    return texte_id, recu, teb, tec


def sauver_wav(enreg, M, texte_id, args):
    os.makedirs("enregistrements", exist_ok=True)
    chemin = (f"enregistrements/{datetime.now():%Y%m%d_%H%M%S}_M{M}_{texte_id or 'libre'}"
              f"_{args.distance}m_{args.bruit}.wav")
    wavfile.write(chemin, FS, enreg.astype(np.float32))
    return chemin


def noter_et_afficher(args, texte_id, recu, teb, tec, chemin_wav):
    nouveau = not os.path.exists(FICHIER_CSV)
    exact = "" if texte_id is None else int(recu == CORPUS[texte_id])
    with open(FICHIER_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        if nouveau:
            w.writerow(["date", "M", "texte", "distance_m", "bruit", "exact", "TEB", "TEC", "recu", "fichier"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.M, texte_id or "libre", args.distance,
                    args.bruit, exact, "" if teb is None else f"{teb:.4f}",
                    "" if tec is None else f"{tec:.4f}", recu, chemin_wav])
    print(f"Reçu   : {recu!r}")
    if texte_id is None:
        print("Texte hors corpus (ou trop abîmé pour être reconnu) : pas de comparaison.")
    else:
        print(f"Attendu: {CORPUS[texte_id]!r}  ({texte_id})")
        print(f"{'EXACT' if recu == CORPUS[texte_id] else 'ERREURS'} — TEB = {teb:.3f}, TEC = {tec:.3f}")


# ======================================================================
#  Écoute continue : détecte le début et la fin de chaque message
# ======================================================================
class Ecouteur:
    """Reçoit le son du micro par blocs de 50 ms.

    - Au repos, il mesure le niveau du bruit ambiant (médiane des 3 dernières secondes).
    - Un message commence quand le niveau dépasse `sensibilite` x le bruit pendant 100 ms.
      On garde aussi les 1,5 s précédentes pour ne pas perdre le chirp.
    - Le message est fini dès que le SIGNAL DE FIN (chirp descendant) est entendu.
      Secours si le micro l'a raté : le niveau médian des 0,8 dernières secondes
      redescend près du bruit, ou la durée maximale d'un message (255 octets) est atteinte.
    """
    BLOC = int(0.05 * FS)

    def __init__(self, M, sensibilite=2.5):
        self.M, self.sens = M, sensibilite
        # on ne garde que la bande utile (chirp 500–5000 Hz + notes) : moins sensible aux bruits graves
        self.sos = butter(4, [400, 5600], btype="bandpass", fs=FS, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.niveaux_repos = deque(maxlen=60)       # 3 s de niveaux au repos
        self.avant = deque(maxlen=30)               # 1,5 s de son gardé avant le déclenchement
        self.capture, self.niveaux_capture = None, None
        self.au_dessus = 0
        k = int(np.log2(M))
        self.duree_max = int((0.6 + 256 * 8 / k * TS + 1.5) * FS)

    def bruit(self):
        return float(np.median(self.niveaux_repos)) if self.niveaux_repos else 1e-4

    def pret(self):
        return len(self.niveaux_repos) >= 20        # 1 s de calibration

    def ajouter(self, bloc):
        """Ajoute un bloc ; renvoie l'enregistrement complet d'un message quand il est terminé."""
        filtre, self.zi = sosfilt(self.sos, bloc, zi=self.zi)
        niveau = float(np.sqrt(np.mean(filtre ** 2)))

        if self.capture is None:                    # --- au repos
            if self.pret() and niveau > self.sens * self.bruit():
                self.au_dessus += 1
            else:
                self.au_dessus = 0
                self.niveaux_repos.append(niveau)
            self.avant.append(bloc)
            if self.au_dessus >= 2:                 # 100 ms au-dessus du seuil : un message arrive
                self.capture = list(self.avant)
                self.niveaux_capture = []
                self.bruit_fige = self.bruit()
                # le signal de fin ne peut pas être dans le son d'avant ni dans le chirp de départ
                self.fin_min = sum(len(b) for b in self.capture) + int(0.5 * FS)
                self.au_dessus = 0
                return "debut"
            return None

        self.capture.append(bloc)                   # --- message en cours
        self.niveaux_capture.append(niveau)
        taille = sum(len(b) for b in self.capture)
        # signal de fin entendu dans les 2 dernières secondes ?
        queue_son = np.concatenate(self.capture[-40:])
        pos = chercher_fin(queue_son) if len(self.niveaux_capture) > 10 else None
        fin_entendue = pos is not None and taille - len(queue_son) + pos >= self.fin_min
        recents = self.niveaux_capture[-16:]
        silence = len(self.niveaux_capture) > 30 and np.median(recents) < 1.6 * self.bruit_fige
        if fin_entendue or silence or taille >= self.duree_max:
            self.raison = "signal de fin" if fin_entendue else ("silence" if silence else "durée maximale")
            enreg = np.concatenate(self.capture)
            self.capture, self.niveaux_capture = None, None
            self.avant.clear()
            return enreg
        return None


def ecouter(args):
    import sounddevice as sd
    file = queue.Queue()

    def rappel(indata, frames, temps, statut):
        file.put(indata[:, 0].copy())

    ec = Ecouteur(args.M, args.sensibilite)
    mode = "plusieurs messages, Ctrl+C pour arrêter" if args.continu else "s'arrête après un message"
    print(f"Écoute en M = {args.M} ({args.distance} m, {args.bruit}) — {mode}. Mesure du bruit ambiant…")
    nb = 0
    with sd.InputStream(samplerate=FS, channels=1, dtype="float32", blocksize=Ecouteur.BLOC, callback=rappel):
        try:
            annonce = False
            while True:
                r = ec.ajouter(file.get())
                if not annonce and ec.pret():
                    print(f"Prêt : en attente d'un message (bruit ambiant = {ec.bruit():.4f}).")
                    annonce = True
                if isinstance(r, str):
                    print("\nMessage détecté, réception…")
                elif r is not None:
                    nb += 1
                    print(f"Fin du message ({ec.raison}, {len(r) / FS:.1f} s enregistrées). Décodage…")
                    texte_id, recu, teb, tec = analyser(r, args.M)
                    chemin = sauver_wav(r, args.M, texte_id, args)
                    if np.max(np.abs(r)) > 0.99:
                        print("ATTENTION saturation : baissez le volume du haut-parleur.")
                    noter_et_afficher(args, texte_id, recu, teb, tec, chemin)
                    if not args.continu:
                        print("Écoute terminée.")
                        return
                    print(f"\n[{nb} message(s) reçu(s)] En attente du suivant…")
        except KeyboardInterrupt:
            print(f"\nArrêt. {nb} message(s) reçu(s), résultats dans {FICHIER_CSV}.")


# ======================================================================
def main():
    sys.stdout.reconfigure(errors="replace")  # console Windows
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["emettre", "ecouter", "boucle", "analyser"])
    p.add_argument("--M", type=int, required=True, choices=[2, 4, 8, 16, 32])
    p.add_argument("--texte", choices=list(CORPUS), help="obligatoire pour emettre et boucle")
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--bruit", default="calme")
    p.add_argument("--sensibilite", type=float, default=2.5,
                   help="ecouter : seuil de détection en multiple du bruit ambiant (plus bas = plus sensible)")
    p.add_argument("--continu", action="store_true",
                   help="ecouter : rester à l'écoute après chaque message (sinon s'arrête après le premier)")
    p.add_argument("--fichier", help="mode analyser : chemin du .wav")
    args = p.parse_args()
    if args.mode in ("emettre", "boucle") and args.texte is None:
        p.error(f"le mode {args.mode} demande --texte (T1 à T5)")

    if args.mode == "ecouter":
        ecouter(args)
        return

    if args.mode == "emettre":
        import sounddevice as sd
        signal, _ = emettre(CORPUS[args.texte], args.M)
        print(f"Émission de {args.texte} en M = {args.M} ({len(signal) / FS:.1f} s)…")
        # silence avant/après : Windows coupe parfois le début ou la fin d'un son joué
        a_jouer = np.concatenate([np.zeros(int(0.3 * FS)), signal, np.zeros(int(0.7 * FS))])
        sd.play(a_jouer.astype(np.float32), FS)
        sd.wait()
        print("Terminé.")
        return

    if args.mode == "boucle":
        import sounddevice as sd
        signal, _ = emettre(CORPUS[args.texte], args.M)
        # un seul PC : joue sur le haut-parleur et enregistre le micro en même temps
        a_jouer = np.concatenate([np.zeros(int(0.5 * FS)), signal, np.zeros(int(1.5 * FS))]).astype(np.float32)
        print(f"Boucle : émission et enregistrement ({len(a_jouer) / FS:.1f} s)…")
        enreg = sd.playrec(a_jouer, samplerate=FS, channels=1, dtype="float32")
        sd.wait()
        enreg = enreg[:, 0]
        if np.max(np.abs(enreg)) > 0.99:
            print("ATTENTION saturation : baissez le volume du haut-parleur.")
        texte_id, recu, teb, tec = analyser(enreg, args.M, args.texte)
        chemin = sauver_wav(enreg, args.M, texte_id, args)
    else:  # analyser
        fs, enreg = wavfile.read(args.fichier)
        enreg = enreg.astype(np.float64)
        if enreg.ndim > 1:
            enreg = enreg[:, 0]
        texte_id, recu, teb, tec = analyser(enreg, args.M, args.texte)
        chemin = args.fichier
    noter_et_afficher(args, texte_id, recu, teb, tec, chemin)


if __name__ == "__main__":
    main()
