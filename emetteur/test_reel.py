"""Tests acoustiques réels du modem M-FSK.

Deux PC : l'un ÉMET (haut-parleur), l'autre REÇOIT (micro).

  PC émetteur :  python test_reel.py emettre  --M 4 --texte T1
  PC récepteur : python test_reel.py recevoir --M 4 --texte T1 --distance 1 --bruit calme

Le récepteur démarre l'enregistrement, puis l'émetteur lance le son dans
les 2 secondes. Le récepteur décode, compare au texte attendu, sauvegarde
l'enregistrement (.wav) et ajoute une ligne dans resultats_reels.csv.

Pour réanalyser un enregistrement déjà sauvegardé :
  python test_reel.py analyser --M 4 --texte T1 --distance 1 --bruit calme --fichier enregistrements/xxx.wav
"""
import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np
from scipy.io import wavfile

from mfsk import FS, emettre, trouver_debut_robuste, detecter_symboles, symboles_vers_bits, bits_vers_texte

CORPUS = {
    "T1": "Bonjour",
    "T2": "Portez ce vieux whisky au juge blond qui fume",
    "T3": "Code 4021 : 75% ok ! (test #3)",
    "T4": "Élève, été, ça va ? Où ça ?",
    "T5": "Ce message de test sert à mesurer la robustesse du modem M-FSK sur une phrase plus longue.",
}
MARGE = 4.0  # secondes d'enregistrement en plus du message (temps pour lancer l'émetteur)
FICHIER_CSV = "resultats_reels.csv"


def analyser(enreg, M, texte):
    """Décode un enregistrement et compare au texte attendu."""
    _, sym_envoyes = emettre(texte, M)
    debut, methode = trouver_debut_robuste(enreg, M)
    print(f"Synchronisation : {methode} (début du message à {debut / FS:.2f} s)")
    sym_recus = detecter_symboles(enreg, M, debut, len(sym_envoyes))
    if len(sym_recus) < len(sym_envoyes):  # enregistrement coupé trop tôt
        sym_recus = np.concatenate([sym_recus, np.zeros(len(sym_envoyes) - len(sym_recus), dtype=int)])
    b_tx = symboles_vers_bits(sym_envoyes, M)
    b_rx = symboles_vers_bits(sym_recus, M)
    recu = bits_vers_texte(b_rx)
    teb = float(np.mean(b_tx != b_rx))
    # taux d'erreur caractère : caractères différents à la même position
    n = max(len(texte), len(recu))
    tec = sum(1 for i in range(n) if i >= len(recu) or i >= len(texte) or recu[i] != texte[i]) / n
    return recu, teb, tec


def noter(args, recu, teb, tec, chemin_wav):
    nouveau = not os.path.exists(FICHIER_CSV)
    with open(FICHIER_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        if nouveau:
            w.writerow(["date", "M", "texte", "distance_m", "bruit", "exact", "TEB", "TEC", "recu", "fichier"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), args.M, args.texte, args.distance,
                    args.bruit, int(recu == CORPUS[args.texte]), f"{teb:.4f}", f"{tec:.4f}", recu, chemin_wav])


def main():
    sys.stdout.reconfigure(errors="replace")  # console Windows
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["emettre", "recevoir", "boucle", "analyser"])
    p.add_argument("--M", type=int, required=True, choices=[2, 4, 8, 16, 32])
    p.add_argument("--texte", required=True, choices=list(CORPUS))
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--bruit", default="calme")
    p.add_argument("--fichier", help="mode analyser : chemin du .wav")
    args = p.parse_args()
    texte = CORPUS[args.texte]
    signal, _ = emettre(texte, args.M)

    if args.mode == "emettre":
        import sounddevice as sd
        print(f"Émission de {args.texte} en M = {args.M} ({len(signal) / FS:.1f} s)…")
        sd.play(signal.astype(np.float32), FS)
        sd.wait()
        print("Terminé.")
        return

    if args.mode in ("recevoir", "boucle"):
        import sounddevice as sd
        if args.mode == "boucle":
            # un seul PC : joue sur le haut-parleur et enregistre le micro en même temps
            a_jouer = np.concatenate([np.zeros(int(0.5 * FS)), signal, np.zeros(int(1.5 * FS))]).astype(np.float32)
            print(f"Boucle : émission et enregistrement ({len(a_jouer) / FS:.1f} s)…")
            enreg = sd.playrec(a_jouer, samplerate=FS, channels=1, dtype="float32")
        else:
            duree = len(signal) / FS + MARGE
            print(f"Enregistrement pendant {duree:.1f} s : lancez l'émetteur MAINTENANT.")
            enreg = sd.rec(int(duree * FS), samplerate=FS, channels=1, dtype="float32")
        sd.wait()
        enreg = enreg[:, 0]
        os.makedirs("enregistrements", exist_ok=True)
        chemin = f"enregistrements/{datetime.now():%Y%m%d_%H%M%S}_M{args.M}_{args.texte}_{args.distance}m_{args.bruit}.wav"
        wavfile.write(chemin, FS, enreg)
        if np.max(np.abs(enreg)) > 0.99:
            print("ATTENTION saturation : baissez le volume du haut-parleur.")
    else:
        fs, enreg = wavfile.read(args.fichier)
        enreg = enreg.astype(np.float64)
        if enreg.ndim > 1:
            enreg = enreg[:, 0]
        chemin = args.fichier

    recu, teb, tec = analyser(enreg, args.M, texte)
    noter(args, recu, teb, tec, chemin)
    print(f"Reçu   : {recu!r}")
    print(f"Attendu: {texte!r}")
    print(f"{'EXACT' if recu == texte else 'ERREURS'} — TEB = {teb:.3f}, TEC = {tec:.3f}")


if __name__ == "__main__":
    main()
