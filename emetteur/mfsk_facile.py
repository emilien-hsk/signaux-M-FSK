"""
mfsk_facile.py - Modem audio M-FSK, version commentée pas à pas.

Ce fichier fait EXACTEMENT la même chose que mfsk.py : un message envoyé
avec l'un est décodé par l'autre. Seule l'écriture change (boucles simples,
noms explicites, beaucoup de commentaires).

L'IDÉE DE LA M-FSK
------------------
On transforme un texte en une suite de "notes" (des sons purs), qu'on
appelle des SYMBOLES. Avec M notes différentes, une note transporte
k = log2(M) bits :
    M = 2  -> 1 bit par note        M = 8  -> 3 bits par note
    M = 4  -> 2 bits par note       M = 16 -> 4 bits par note
Le symbole numéro s (0 <= s < M) est joué à la fréquence F0 + s * DF.
Exemple avec M = 4 : 1000 Hz, 1100 Hz, 1200 Hz, 1300 Hz.

CE QUE L'ÉMETTEUR JOUE (la "trame")
-----------------------------------
  [préambule][silence][octet de longueur][octets du message][silence][signal de fin]

  - préambule      : son qui MONTE de 500 à 5000 Hz -> "un message commence ici"
  - octet longueur : nombre d'octets du message     -> "il y a tant de notes à lire"
  - signal de fin  : son qui DESCEND de 5000 à 500 Hz -> "le message est fini"
"""
import numpy as np
from scipy.signal import correlate

# =====================================================================
#  PARAMÈTRES (les mêmes côté émetteur et côté récepteur !)
# =====================================================================
FS = 44100      # fréquence d'échantillonnage : 44 100 valeurs par seconde
TS = 0.100      # durée d'une note : 100 ms
F0 = 1000       # fréquence du symbole 0 (Hz)
DF = 100        # écart entre deux symboles voisins (Hz)
PRE_T = 0.5     # durée du préambule (s)
GAP_T = 0.1     # durée des silences autour des notes (s)
FIN_T = 0.3     # durée du signal de fin (s)
AMP = 0.8       # volume (1.0 = maximum)

N_NOTE = int(TS * FS)       # nombre d'échantillons dans une note (4410)


# =====================================================================
#  PETITS CALCULS UTILES
# =====================================================================
def bits_par_symbole(M):
    """Nombre de bits transportés par un symbole : k = log2(M)."""
    return int(np.log2(M))


def frequences(M):
    """Liste des M fréquences utilisées : [F0, F0 + DF, F0 + 2*DF, ...]."""
    return [F0 + DF * s for s in range(M)]


def nb_symboles_pour(nb_octets, M):
    """Nombre de symboles nécessaires pour envoyer nb_octets octets.
    Division arrondie vers le HAUT : s'il reste quelques bits, il faut une note de plus."""
    k = bits_par_symbole(M)
    nb_bits = 8 * nb_octets
    return (nb_bits + k - 1) // k


# =====================================================================
#  1) TEXTE <-> OCTETS <-> BITS <-> SYMBOLES
# =====================================================================
def texte_vers_octets(texte):
    """'Bonjour' -> [7, 'B', 'o', 'n', ...] : l'octet de longueur puis le texte en UTF-8."""
    message = texte.encode("utf-8")
    if len(message) > 255:
        raise ValueError("message trop long (255 octets max)")
    return bytes([len(message)]) + message


def octets_vers_bits(octets):
    """Chaque octet devient 8 bits, du bit de poids fort au bit de poids faible.
    Exemple : 66 ('B') -> 0 1 0 0 0 0 1 0"""
    bits = []
    for octet in octets:
        for position in range(7, -1, -1):          # 7, 6, 5, ..., 0
            bits.append((octet >> position) & 1)   # on isole le bit à cette position
    return bits


def bits_vers_octets(bits):
    """Inverse : on regroupe les bits par 8. Les bits en trop à la fin sont ignorés."""
    octets = []
    for debut in range(0, len(bits) - 7, 8):
        valeur = 0
        for bit in bits[debut:debut + 8]:
            valeur = valeur * 2 + int(bit)         # comme lire un nombre binaire de gauche à droite
        octets.append(valeur)
    return bytes(octets)


def bits_vers_symboles(bits, M):
    """On regroupe les bits par paquets de k : chaque paquet est un numéro de symbole.
    Exemple avec M = 4 (k = 2) : 01 00 00 10 -> 1, 0, 0, 2"""
    k = bits_par_symbole(M)
    bits = list(bits)
    while len(bits) % k != 0:                      # on complète avec des 0 si besoin
        bits.append(0)
    symboles = []
    for debut in range(0, len(bits), k):
        valeur = 0
        for bit in bits[debut:debut + k]:
            valeur = valeur * 2 + int(bit)
        symboles.append(valeur)
    return symboles


def symboles_vers_bits(symboles, M):
    """Inverse : chaque symbole redevient k bits."""
    k = bits_par_symbole(M)
    bits = []
    for s in symboles:
        for position in range(k - 1, -1, -1):
            bits.append((int(s) >> position) & 1)
    return bits


def octets_vers_texte(octets):
    """Le premier octet donne la longueur ; on décode ensuite ce nombre d'octets en UTF-8.
    errors="replace" : un octet abîmé devient '?' au lieu de faire planter le programme."""
    if len(octets) == 0:
        return ""
    longueur = octets[0]
    return octets[1:1 + longueur].decode("utf-8", errors="replace")


def texte_vers_symboles(texte, M):
    """Toute la chaîne côté émetteur : texte -> octets -> bits -> symboles."""
    return bits_vers_symboles(octets_vers_bits(texte_vers_octets(texte)), M)


def symboles_vers_texte(symboles, M):
    """Toute la chaîne côté récepteur : symboles -> bits -> octets -> texte."""
    return octets_vers_texte(bits_vers_octets(symboles_vers_bits(symboles, M)))


# =====================================================================
#  2) FABRIQUER LES SONS
# =====================================================================
def instants(duree):
    """Les instants t (en secondes) de chaque échantillon d'un son de cette durée."""
    nb_echantillons = int(duree * FS)
    return np.arange(nb_echantillons) / FS


def chirp(f_debut, f_fin, duree):
    """Son dont la fréquence glisse en ligne droite de f_debut à f_fin.
    La phase vaut 2*pi*(f_debut*t + vitesse*t^2/2) : sa dérivée donne bien
    une fréquence qui augmente (ou diminue) régulièrement.
    np.hanning fait monter puis redescendre le volume en douceur."""
    t = instants(duree)
    vitesse = (f_fin - f_debut) / duree                 # en Hz par seconde
    phase = 2 * np.pi * (f_debut * t + vitesse * t ** 2 / 2)
    return AMP * np.cos(phase) * np.hanning(len(t))


def preambule():
    """Début de message : chirp MONTANT 500 -> 5000 Hz."""
    return chirp(500, 5000, PRE_T)


def signal_fin():
    """Fin de message : chirp DESCENDANT 5000 -> 500 Hz (facile à distinguer du préambule)."""
    return chirp(5000, 500, FIN_T)


def silence(duree):
    return np.zeros(int(duree * FS))


def note(s, M):
    """Le son du symbole s : une sinusoïde de 100 ms à la fréquence F0 + s*DF.
    On ajoute un fondu de 5 ms au début et à la fin pour éviter les 'clics'."""
    t = instants(TS)
    son = AMP * np.sin(2 * np.pi * frequences(M)[s] * t)

    nb_fondu = int(0.005 * FS)
    volume = np.ones(len(t))
    volume[:nb_fondu] = np.linspace(0, 1, nb_fondu)    # le volume monte de 0 à 1
    volume[-nb_fondu:] = np.linspace(1, 0, nb_fondu)   # puis redescend de 1 à 0
    return son * volume


# =====================================================================
#  3) ÉMETTEUR
# =====================================================================
def emettre(texte, M):
    """Fabrique tout le son à jouer. Renvoie (son, liste des symboles envoyés)."""
    symboles = texte_vers_symboles(texte, M)
    morceaux = [preambule(), silence(GAP_T)]
    for s in symboles:
        morceaux.append(note(s, M))
    morceaux.append(silence(GAP_T))
    morceaux.append(signal_fin())
    return np.concatenate(morceaux), symboles


# =====================================================================
#  4) RÉCEPTEUR
# =====================================================================
def trouver_debut(signal):
    """Trouve où commencent les notes grâce au préambule.

    CORRÉLATION : on fait glisser le préambule (qu'on connaît) le long du
    signal reçu, et on mesure à chaque position à quel point ils se
    ressemblent. La position la plus ressemblante est celle du préambule.
    Les notes commencent après le préambule et le silence."""
    pre = preambule()
    ressemblance = np.abs(correlate(signal, pre, mode="valid", method="fft"))
    position_preambule = int(np.argmax(ressemblance))
    return position_preambule + len(pre) + int(GAP_T * FS)


def chercher_fin(signal, seuil=30.0):
    """Cherche le signal de fin. Renvoie la position où il commence, ou None s'il n'y est pas.
    On fait la même corrélation que pour le préambule, mais on vérifie en plus
    que le pic est bien réel (et pas juste du bruit) :
      - il doit être 30 fois plus haut que le niveau habituel (la médiane) ;
      - la ressemblance "normalisée" (entre 0 et 1) doit dépasser 0,15."""
    fin = signal_fin()
    if len(signal) < len(fin):
        return None
    ressemblance = np.abs(correlate(signal, fin, mode="valid", method="fft"))
    pic = int(np.argmax(ressemblance))

    if ressemblance[pic] < seuil * (np.median(ressemblance) + 1e-12):
        return None                       # pic pas assez net : c'est du bruit

    morceau = signal[pic:pic + len(fin)]
    ressemblance_normalisee = ressemblance[pic] / (np.linalg.norm(fin) * np.linalg.norm(morceau) + 1e-12)
    if ressemblance_normalisee < 0.15:
        return None                       # ça ne ressemble pas assez au chirp descendant
    return pic


def energies_des_symboles(morceau, M):
    """Pour un morceau de N_NOTE échantillons : l'énergie à chacune des M fréquences.

    La FFT découpe le son en fréquences. Avec 4410 échantillons, elle donne
    une "case" tous les FS / N_NOTE = 10 Hz : la fréquence f est dans la case f / 10.
    La fenêtre de Hann adoucit les bords du morceau pour une FFT plus propre."""
    spectre = np.abs(np.fft.rfft(morceau * np.hanning(N_NOTE)))
    energies = []
    for f in frequences(M):
        case = round(f * N_NOTE / FS)
        energies.append(spectre[case])
    return np.array(energies)


def detecter_symboles(signal, M, debut, nb=None):
    """Lit les notes une par une à partir de 'debut'.
    Pour chaque note : le symbole est la fréquence la plus forte.
    nb = nombre de notes à lire (None = jusqu'au bout du signal)."""
    symboles = []
    position = debut
    while position + N_NOTE <= len(signal):
        if nb is not None and len(symboles) >= nb:
            break
        energies = energies_des_symboles(signal[position:position + N_NOTE], M)
        symboles.append(int(np.argmax(energies)))
        position += N_NOTE
    return symboles


def lire_longueur(signal, M, debut):
    """Lit l'octet de longueur (les toutes premières notes). None si le signal est trop court."""
    nb = nb_symboles_pour(1, M)
    entete = detecter_symboles(signal, M, debut, nb)
    if len(entete) < nb:
        return None
    huit_bits = symboles_vers_bits(entete, M)[:8]
    return bits_vers_octets(huit_bits)[0]


def recevoir(signal, M):
    """Décode un signal enregistré. Renvoie (texte, liste des symboles reçus)."""
    debut = trouver_debut(signal)
    longueur = lire_longueur(signal, M, debut)       # 1) combien d'octets ?
    if longueur is None:
        return "", []
    nb = nb_symboles_pour(1 + longueur, M)           # 2) on lit tout : longueur + message
    symboles = detecter_symboles(signal, M, debut, nb)
    return symboles_vers_texte(symboles, M), symboles


# =====================================================================
#  5) SYNCHRONISATION DE SECOURS
#
#  Pendant les tests réels, le micro coupait parfois le début du son
#  (donc le préambule) : le récepteur se calait au mauvais endroit et
#  tout le message était faux. Ces fonctions retrouvent le début des
#  notes autrement, en regardant les notes elles-mêmes.
# =====================================================================
def nettete(signal, M, debut, nb):
    """Mesure si on est bien calé sur les notes quand on commence à 'debut'.
    Pour chaque note : énergie de la fréquence la plus forte / énergie des M fréquences.
      - proche de 1 : une seule fréquence ressort -> la fenêtre tombe pile sur une note ;
      - plus petit  : la fenêtre est à cheval sur deux notes -> mal calé.
    On additionne ce score sur nb notes (le maximum vaut donc nb)."""
    total = 0.0
    for i in range(nb):
        a = debut + i * N_NOTE
        morceau = signal[a:a + N_NOTE]
        if len(morceau) < N_NOTE:
            break
        energies = energies_des_symboles(morceau, M)
        total += energies.max() / (energies.sum() + 1e-12)
    return total


def zone_de_son(signal):
    """Où le son commence et où les notes finissent : renvoie (debut, fin), ou None.
      1) on découpe le signal en tranches de 10 ms et on mesure le volume de chacune ;
      2) bruit = volume des tranches calmes, fort = volume des tranches les plus fortes ;
      3) une tranche "a du son" si son volume dépasse bruit + 20 % de l'écart ;
      4) si on entend le signal de fin, les notes s'arrêtent juste avant lui."""
    pas = int(0.01 * FS)
    volumes = []
    for a in range(0, len(signal) - pas, pas):
        tranche = signal[a:a + pas]
        volumes.append(np.sqrt(np.mean(tranche ** 2)))   # volume "RMS"
    if len(volumes) == 0:
        return None

    bruit = np.percentile(volumes, 10)    # 10 % des tranches sont plus calmes que ça
    fort = np.percentile(volumes, 99)     # 1 % des tranches sont plus fortes que ça
    seuil = bruit + 0.2 * (fort - bruit)
    avec_son = [i for i in range(len(volumes)) if volumes[i] > seuil]
    if len(avec_son) == 0:
        return None

    debut = avec_son[0] * pas
    fin = (avec_son[-1] + 1) * pas
    position_fin = chercher_fin(signal)
    if position_fin is not None and position_fin > debut:
        fin = position_fin - int(GAP_T * FS)
    return debut, fin


def fin_annoncee(signal, M, debut):
    """Si le message commence en 'debut', où doit-il finir ? (calculé avec l'octet de longueur)"""
    longueur = lire_longueur(signal, M, debut)
    if longueur is None:
        return None
    return debut + nb_symboles_pour(1 + longueur, M) * N_NOTE


def synchro_par_notes(signal, M):
    """Trouve le début des notes SANS utiliser le préambule. None si impossible."""
    zone = zone_de_son(signal)
    if zone is None:
        return None
    debut_son, fin_son = zone

    # Étape 1 - calage fin : on essaie tous les départs possibles sur la durée
    # d'une note, tous les 2 ms, et on garde celui où les 6 premières notes sont les plus nettes.
    calage = debut_son
    meilleure_nettete = -1
    for d in range(debut_son, debut_son + N_NOTE, int(0.002 * FS)):
        n = nettete(signal, M, d, 6)
        if n > meilleure_nettete:
            meilleure_nettete, calage = n, d

    # Étape 2 - on est calé sur UNE note, mais peut-être pas sur la PREMIÈRE.
    # On essaie de 5 notes avant à 2 notes après. Le bon départ est celui dont
    # l'octet de longueur annonce une fin proche de la vraie fin du son.
    nb_entete = nb_symboles_pour(1, M)
    meilleur_debut = None
    plus_petit_ecart = None
    for decalage in range(-5, 3):
        d = calage + decalage * N_NOTE
        if d < 0:
            continue
        if nettete(signal, M, d, nb_entete) < 0.5 * nb_entete:
            continue                       # pas de vraies notes ici (silence ou bruit)
        fin_prevue = fin_annoncee(signal, M, d)
        if fin_prevue is None:
            continue
        ecart = abs(fin_prevue - fin_son)
        if plus_petit_ecart is None or ecart < plus_petit_ecart:
            meilleur_debut, plus_petit_ecart = d, ecart
    return meilleur_debut


def trouver_debut_robuste(signal, M):
    """Choisit le début du message. Renvoie (debut, méthode utilisée).
      1) le préambule, si l'octet de longueur lu à cet endroit annonce bien la vraie fin du son ;
      2) sinon la synchro par les notes, si elle est cohérente ;
      3) sinon (octet de longueur abîmé) : celle des deux où les notes sont les plus nettes."""
    zone = zone_de_son(signal)
    debut_preambule = trouver_debut(signal)
    if zone is None:
        return debut_preambule, "chirp"
    fin_son = zone[1]

    def coherent(debut):
        fin_prevue = fin_annoncee(signal, M, debut)
        return fin_prevue is not None and abs(fin_prevue - fin_son) < N_NOTE

    if coherent(debut_preambule):
        return debut_preambule, "chirp"

    debut_notes = synchro_par_notes(signal, M)
    if debut_notes is not None and coherent(debut_notes):
        return debut_notes, "notes"

    if debut_notes is not None and nettete(signal, M, debut_notes, 10) > nettete(signal, M, debut_preambule, 10) + 0.5:
        return debut_notes, "notes (non confirmé)"
    return debut_preambule, "chirp (non confirmé)"
