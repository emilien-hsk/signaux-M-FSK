"""
test_reel_facile.py - Tests acoustiques réels du modem M-FSK (version commentée).

Fait la même chose que test_reel.py, avec les mêmes commandes, et utilise mfsk_facile.py.

PC émetteur (haut-parleur) : joue un texte du corpus
    python test_reel_facile.py emettre --M 4 --texte T1

PC récepteur (micro) : écoute sans savoir quel texte arrive
    python test_reel_facile.py ecouter --M 4 --distance 1 --bruit calme
    (ajouter --continu pour recevoir plusieurs messages, Ctrl+C pour arrêter)

Un seul PC + un baffle (joue et enregistre en même temps) :
    python test_reel_facile.py boucle --M 4 --texte T1 --distance 1 --bruit calme

Réanalyser un enregistrement sauvegardé :
    python test_reel_facile.py analyser --M 4 --distance 1 --bruit calme --fichier enregistrements/xxx.wav
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
from scipy.signal import butter, sosfilt

import mfsk_facile as mfsk

# Les 5 textes de test.
CORPUS = {
    "T1": "Bonjour",
    "T2": "Portez ce vieux whisky au juge blond qui fume",
    "T3": "Code 4021 : 75% ok ! (test #3)",
    "T4": "Élève, été, ça va ? Où ça ?",
    "T5": "Ce message de test sert à mesurer la robustesse du modem M-FSK sur une phrase plus longue.",
}

# Les 5 textes ont des longueurs différentes : la longueur suffit pour savoir lequel est arrivé.
# TEXTE_PAR_LONGUEUR = {7: "T1", 45: "T2", ...}
TEXTE_PAR_LONGUEUR = {}
for identifiant, texte in CORPUS.items():
    TEXTE_PAR_LONGUEUR[len(texte.encode("utf-8"))] = identifiant

FICHIER_CSV = "resultats_reels.csv"


# =====================================================================
#  MESURES D'ERREUR
# =====================================================================
def taux_erreur_bits(bits_envoyes, bits_recus):
    """TEB : proportion de bits faux (0 = parfait, 0.5 = comme au hasard)."""
    nb_faux = 0
    for b_envoye, b_recu in zip(bits_envoyes, bits_recus):
        if b_envoye != b_recu:
            nb_faux += 1
    return nb_faux / len(bits_envoyes)


def taux_erreur_caracteres(attendu, recu):
    """TEC : proportion de caractères faux (un caractère manquant compte comme faux)."""
    n = max(len(attendu), len(recu))
    nb_faux = 0
    for i in range(n):
        if i >= len(attendu) or i >= len(recu) or attendu[i] != recu[i]:
            nb_faux += 1
    return nb_faux / n


# =====================================================================
#  ANALYSE D'UN ENREGISTREMENT
# =====================================================================
def identifier(enreg, M, debut):
    """Devine quel texte du corpus a été envoyé.
    Renvoie (identifiant ou None, texte décodé)."""
    longueur = mfsk.lire_longueur(enreg, M, debut)
    if longueur is None:
        return None, ""
    symboles = mfsk.detecter_symboles(enreg, M, debut, mfsk.nb_symboles_pour(1 + longueur, M))
    recu = mfsk.symboles_vers_texte(symboles, M)

    # Cas simple : l'octet de longueur correspond à un texte du corpus.
    if longueur in TEXTE_PAR_LONGUEUR:
        return TEXTE_PAR_LONGUEUR[longueur], recu

    # Plan B : l'octet de longueur est abîmé. On compare les bits reçus avec ceux
    # de chaque texte du corpus (sans l'octet de longueur) et on garde le plus proche.
    nb_entete = mfsk.nb_symboles_pour(1, M)
    meilleur_id = None
    meilleur_teb = None
    for identifiant, texte in CORPUS.items():
        sym_attendus = mfsk.texte_vers_symboles(texte, M)
        sym_recus = mfsk.detecter_symboles(enreg, M, debut, len(sym_attendus))
        if len(sym_recus) < len(sym_attendus):
            continue                                   # enregistrement trop court pour ce texte
        teb = taux_erreur_bits(mfsk.symboles_vers_bits(sym_attendus[nb_entete:], M),
                               mfsk.symboles_vers_bits(sym_recus[nb_entete:], M))
        if meilleur_teb is None or teb < meilleur_teb:
            meilleur_id, meilleur_teb = identifiant, teb

    if meilleur_teb is not None and meilleur_teb < 0.2:   # au hasard, on aurait environ 50 %
        return meilleur_id, recu
    return None, recu                                     # texte libre, hors corpus


def analyser(enreg, M, texte_id=None):
    """Décode un enregistrement. Renvoie (texte_id, reçu, TEB, TEC).
    Si texte_id n'est pas donné, on le devine. TEB et TEC valent None si on n'y arrive pas."""
    debut, methode = mfsk.trouver_debut_robuste(enreg, M)
    print(f"Synchronisation : {methode} (début du message à {debut / mfsk.FS:.2f} s)")

    if texte_id is None:
        texte_id, recu = identifier(enreg, M, debut)
        if texte_id is None:
            return None, recu, None, None

    texte = CORPUS[texte_id]
    sym_envoyes = mfsk.texte_vers_symboles(texte, M)
    sym_recus = mfsk.detecter_symboles(enreg, M, debut, len(sym_envoyes))
    while len(sym_recus) < len(sym_envoyes):       # enregistrement coupé trop tôt : on complète
        sym_recus.append(0)

    bits_envoyes = mfsk.symboles_vers_bits(sym_envoyes, M)
    bits_recus = mfsk.symboles_vers_bits(sym_recus, M)
    recu = mfsk.symboles_vers_texte(sym_recus, M)
    return texte_id, recu, taux_erreur_bits(bits_envoyes, bits_recus), taux_erreur_caracteres(texte, recu)


# =====================================================================
#  SAUVEGARDE DES RÉSULTATS
# =====================================================================
def sauver_wav(enreg, M, texte_id, args):
    """Sauvegarde le son capté dans enregistrements/ et renvoie le chemin du fichier."""
    os.makedirs("enregistrements", exist_ok=True)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    nom_texte = texte_id if texte_id is not None else "libre"
    chemin = f"enregistrements/{date}_M{M}_{nom_texte}_{args.distance}m_{args.bruit}.wav"
    wavfile.write(chemin, mfsk.FS, enreg.astype(np.float32))
    return chemin


def noter_et_afficher(args, texte_id, recu, teb, tec, chemin_wav):
    """Ajoute une ligne dans le fichier CSV et affiche le résultat."""
    fichier_existe = os.path.exists(FICHIER_CSV)
    if texte_id is None:
        exact, nom_texte, teb_txt, tec_txt = "", "libre", "", ""
    else:
        exact = 1 if recu == CORPUS[texte_id] else 0
        nom_texte, teb_txt, tec_txt = texte_id, f"{teb:.4f}", f"{tec:.4f}"

    with open(FICHIER_CSV, "a", newline="", encoding="utf-8") as f:
        ecrivain = csv.writer(f, delimiter=";")
        if not fichier_existe:
            ecrivain.writerow(["date", "M", "texte", "distance_m", "bruit", "exact", "TEB", "TEC", "recu", "fichier"])
        ecrivain.writerow([datetime.now().isoformat(timespec="seconds"), args.M, nom_texte, args.distance,
                           args.bruit, exact, teb_txt, tec_txt, recu, chemin_wav])

    print(f"Reçu   : {recu!r}")
    if texte_id is None:
        print("Texte hors corpus (ou trop abîmé pour être reconnu) : pas de comparaison.")
    else:
        print(f"Attendu: {CORPUS[texte_id]!r}  ({texte_id})")
        verdict = "EXACT" if exact == 1 else "ERREURS"
        print(f"{verdict} - TEB = {teb:.3f}, TEC = {tec:.3f}")


def verifier_saturation(enreg):
    if np.max(np.abs(enreg)) > 0.99:
        print("ATTENTION saturation : baissez le volume du haut-parleur.")


# =====================================================================
#  ÉCOUTE EN CONTINU
# =====================================================================
def nb_echantillons(blocs):
    return sum(len(b) for b in blocs)


class Ecouteur:
    """Reçoit le son du micro par petits blocs de 50 ms et repère les messages.

    Il a deux états :
      AU REPOS      : il mesure le bruit ambiant. Si le volume dépasse
                      'sensibilite' x le bruit pendant 2 blocs (100 ms),
                      un message commence. On garde aussi les 1,5 s
                      d'avant pour ne pas perdre le préambule.
      EN RÉCEPTION  : il accumule les blocs jusqu'à la fin du message :
                      - normalement quand il entend le signal de fin ;
                      - en secours : quand le calme revient, ou si le
                        message dure plus longtemps que le maximum possible.
    """
    TAILLE_BLOC = int(0.05 * mfsk.FS)

    def __init__(self, M, sensibilite=2.5):
        self.M = M
        self.sensibilite = sensibilite

        # Filtre passe-bande 400-5600 Hz : on ne garde que les fréquences du modem,
        # pour être moins gêné par les bruits graves (voix, ventilateur...).
        self.filtre = butter(4, [400, 5600], btype="bandpass", fs=mfsk.FS, output="sos")
        self.memoire_filtre = np.zeros((self.filtre.shape[0], 2))   # le filtre se souvient du bloc précédent

        self.volumes_repos = deque(maxlen=60)   # volumes des 3 dernières secondes au repos
        self.blocs_avant = deque(maxlen=30)     # son des 1,5 dernières secondes
        self.nb_blocs_forts = 0

        self.en_reception = False
        self.blocs_message = []
        self.volumes_message = []
        self.bruit_au_debut = 0.0
        self.fin_minimum = 0
        self.raison_fin = ""

        # Durée max d'un message : préambule + 255 octets (+ longueur) + marge.
        k = mfsk.bits_par_symbole(M)
        self.duree_max = int((0.6 + 256 * 8 / k * mfsk.TS + 1.5) * mfsk.FS)

    def bruit(self):
        """Niveau du bruit ambiant : médiane des volumes au repos."""
        if len(self.volumes_repos) == 0:
            return 1e-4
        return float(np.median(self.volumes_repos))

    def pret(self):
        """Prêt après 1 s de mesure du bruit (20 blocs)."""
        return len(self.volumes_repos) >= 20

    def ajouter(self, bloc):
        """Donne un nouveau bloc de son à l'écouteur. Renvoie (événement, enregistrement) :
            ("rien", None)   -> rien de spécial
            ("debut", None)  -> un message vient de commencer
            ("fin", enreg)   -> le message est fini, voici tout le son capté"""
        bloc_filtre, self.memoire_filtre = sosfilt(self.filtre, bloc, zi=self.memoire_filtre)
        volume = float(np.sqrt(np.mean(bloc_filtre ** 2)))
        if self.en_reception:
            return self.continuer_reception(bloc, volume)
        return self.attendre_message(bloc, volume)

    def attendre_message(self, bloc, volume):
        if self.pret() and volume > self.sensibilite * self.bruit():
            self.nb_blocs_forts += 1
        else:
            self.nb_blocs_forts = 0
            self.volumes_repos.append(volume)    # bloc calme : il sert à mesurer le bruit
        self.blocs_avant.append(bloc)

        if self.nb_blocs_forts < 2:
            return "rien", None

        # 100 ms de son fort : un message commence !
        self.en_reception = True
        self.blocs_message = list(self.blocs_avant)
        self.volumes_message = []
        self.bruit_au_debut = self.bruit()
        # Le signal de fin ne peut pas être dans le son d'avant, ni dans le préambule :
        # on ne l'accepte qu'au moins 0,5 s après le déclenchement.
        self.fin_minimum = nb_echantillons(self.blocs_message) + int(0.5 * mfsk.FS)
        self.nb_blocs_forts = 0
        return "debut", None

    def continuer_reception(self, bloc, volume):
        self.blocs_message.append(bloc)
        self.volumes_message.append(volume)
        taille = nb_echantillons(self.blocs_message)

        # 1) Le signal de fin est-il dans les 2 dernières secondes (40 blocs) ?
        fin_entendue = False
        if len(self.volumes_message) > 10:
            derniers = np.concatenate(self.blocs_message[-40:])
            position = mfsk.chercher_fin(derniers)
            if position is not None:
                position_dans_message = taille - len(derniers) + position
                fin_entendue = position_dans_message >= self.fin_minimum

        # 2) Secours : le calme est revenu (volume médian des 0,8 dernières secondes proche du bruit).
        retour_au_calme = False
        if len(self.volumes_message) > 30:
            retour_au_calme = np.median(self.volumes_message[-16:]) < 1.6 * self.bruit_au_debut

        # 3) Secours : plus long que le plus long message possible.
        trop_long = taille >= self.duree_max

        if not (fin_entendue or retour_au_calme or trop_long):
            return "rien", None

        if fin_entendue:
            self.raison_fin = "signal de fin"
        elif retour_au_calme:
            self.raison_fin = "silence"
        else:
            self.raison_fin = "durée maximale"
        enregistrement = np.concatenate(self.blocs_message)

        # Retour à l'état de repos
        self.en_reception = False
        self.blocs_message = []
        self.volumes_message = []
        self.blocs_avant.clear()
        return "fin", enregistrement


def mode_ecouter(args):
    import sounddevice as sd

    # Le micro appelle 'rappel' tous les 50 ms avec un nouveau bloc de son.
    # On le range dans une file d'attente ; la boucle principale les traite un par un.
    file_attente = queue.Queue()

    def rappel(indata, frames, temps, statut):
        file_attente.put(indata[:, 0].copy())

    ecouteur = Ecouteur(args.M, args.sensibilite)
    if args.continu:
        mode = "plusieurs messages, Ctrl+C pour arrêter"
    else:
        mode = "s'arrête après un message"
    print(f"Écoute en M = {args.M} ({args.distance} m, {args.bruit}) - {mode}. Mesure du bruit ambiant…")

    nb_recus = 0
    deja_annonce = False
    with sd.InputStream(samplerate=mfsk.FS, channels=1, dtype="float32",
                        blocksize=Ecouteur.TAILLE_BLOC, callback=rappel):
        try:
            while True:
                evenement, enreg = ecouteur.ajouter(file_attente.get())

                if not deja_annonce and ecouteur.pret():
                    print(f"Prêt : en attente d'un message (bruit ambiant = {ecouteur.bruit():.4f}).")
                    deja_annonce = True

                if evenement == "debut":
                    print("\nMessage détecté, réception…")

                elif evenement == "fin":
                    nb_recus += 1
                    print(f"Fin du message ({ecouteur.raison_fin}, {len(enreg) / mfsk.FS:.1f} s enregistrées). Décodage…")
                    texte_id, recu, teb, tec = analyser(enreg, args.M)
                    chemin = sauver_wav(enreg, args.M, texte_id, args)
                    verifier_saturation(enreg)
                    noter_et_afficher(args, texte_id, recu, teb, tec, chemin)
                    if not args.continu:
                        print("Écoute terminée.")
                        return
                    print(f"\n[{nb_recus} message(s) reçu(s)] En attente du suivant…")
        except KeyboardInterrupt:
            print(f"\nArrêt. {nb_recus} message(s) reçu(s), résultats dans {FICHIER_CSV}.")


# =====================================================================
#  LES AUTRES MODES
# =====================================================================
def mode_emettre(args):
    import sounddevice as sd
    son, _ = mfsk.emettre(CORPUS[args.texte], args.M)
    print(f"Émission de {args.texte} en M = {args.M} ({len(son) / mfsk.FS:.1f} s)…")
    # Silence avant et après : Windows coupe parfois le début ou la fin d'un son joué.
    a_jouer = np.concatenate([mfsk.silence(0.3), son, mfsk.silence(0.7)])
    sd.play(a_jouer.astype(np.float32), mfsk.FS)
    sd.wait()
    print("Terminé.")


def mode_boucle(args):
    """Un seul PC : joue sur le haut-parleur et enregistre le micro en même temps."""
    import sounddevice as sd
    son, _ = mfsk.emettre(CORPUS[args.texte], args.M)
    a_jouer = np.concatenate([mfsk.silence(0.5), son, mfsk.silence(1.5)]).astype(np.float32)
    print(f"Boucle : émission et enregistrement ({len(a_jouer) / mfsk.FS:.1f} s)…")
    enreg = sd.playrec(a_jouer, samplerate=mfsk.FS, channels=1, dtype="float32")
    sd.wait()
    enreg = enreg[:, 0]                       # on garde le seul canal (mono)
    verifier_saturation(enreg)
    texte_id, recu, teb, tec = analyser(enreg, args.M, args.texte)
    chemin = sauver_wav(enreg, args.M, texte_id, args)
    noter_et_afficher(args, texte_id, recu, teb, tec, chemin)


def mode_analyser(args):
    _, enreg = wavfile.read(args.fichier)
    enreg = enreg.astype(np.float64)
    if enreg.ndim > 1:                        # fichier stéréo : on garde le canal de gauche
        enreg = enreg[:, 0]
    texte_id, recu, teb, tec = analyser(enreg, args.M, args.texte)
    noter_et_afficher(args, texte_id, recu, teb, tec, args.fichier)


# =====================================================================
#  PROGRAMME PRINCIPAL
# =====================================================================
def main():
    sys.stdout.reconfigure(errors="replace")  # évite les plantages d'affichage dans la console Windows

    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["emettre", "ecouter", "boucle", "analyser"])
    p.add_argument("--M", type=int, required=True, choices=[2, 4, 8, 16, 32])
    p.add_argument("--texte", choices=list(CORPUS), help="obligatoire pour emettre et boucle")
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--bruit", default="calme")
    p.add_argument("--sensibilite", type=float, default=2.5,
                   help="ecouter : seuil de détection en multiple du bruit ambiant (plus bas = plus sensible)")
    p.add_argument("--continu", action="store_true",
                   help="ecouter : rester à l'écoute après chaque message")
    p.add_argument("--fichier", help="mode analyser : chemin du .wav")
    args = p.parse_args()

    if args.mode in ("emettre", "boucle") and args.texte is None:
        p.error(f"le mode {args.mode} demande --texte (T1 à T5)")

    if args.mode == "emettre":
        mode_emettre(args)
    elif args.mode == "ecouter":
        mode_ecouter(args)
    elif args.mode == "boucle":
        mode_boucle(args)
    else:
        mode_analyser(args)


if __name__ == "__main__":
    main()
