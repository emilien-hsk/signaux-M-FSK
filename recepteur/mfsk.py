"""Modem audio M-FSK : émetteur et récepteur.

Paramètres (fixés en octobre) :
    fs = 44 100 Hz, Ts = 100 ms, f0 = 1000 Hz, pas = 100 Hz,
    préambule = chirp 500 -> 5000 Hz pendant 0,5 s.

Trame émise : [chirp][silence 0,1 s][1 octet = longueur][message UTF-8]
"""
import numpy as np
from scipy.signal import chirp, correlate

FS = 44100
TS = 0.100
F0 = 1000
DF = 100
PRE_T = 0.5
GAP_T = 0.1
AMP = 0.8


def preambule():
    t = np.arange(int(PRE_T * FS)) / FS
    return AMP * chirp(t, f0=500, t1=PRE_T, f1=5000) * np.hanning(len(t))


def frequences(M):
    return F0 + DF * np.arange(M)


# ---------- texte <-> bits <-> symboles ----------
def texte_vers_bits(texte):
    data = texte.encode("utf-8")
    if len(data) > 255:
        raise ValueError("message trop long (255 octets max)")
    octets = bytes([len(data)]) + data
    return np.unpackbits(np.frombuffer(octets, dtype=np.uint8))


def bits_vers_symboles(bits, M):
    k = int(np.log2(M))
    reste = (-len(bits)) % k
    bits = np.concatenate([bits, np.zeros(reste, dtype=np.uint8)])
    groupes = bits.reshape(-1, k)
    poids = 2 ** np.arange(k - 1, -1, -1)
    return groupes @ poids


def symboles_vers_bits(symboles, M):
    k = int(np.log2(M))
    return ((np.asarray(symboles)[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def bits_vers_texte(bits):
    n = len(bits) // 8 * 8
    octets = np.packbits(bits[:n]).tobytes()
    if not octets:
        return ""
    longueur = octets[0]
    return octets[1:1 + longueur].decode("utf-8", errors="replace")


# ---------- émetteur ----------
def emettre(texte, M):
    sym = bits_vers_symboles(texte_vers_bits(texte), M)
    n = int(TS * FS)
    t = np.arange(n) / FS
    rampe = np.ones(n)
    r = int(0.005 * FS)  # 5 ms de fondu pour éviter les clics
    rampe[:r] = np.linspace(0, 1, r)
    rampe[-r:] = np.linspace(1, 0, r)
    f = frequences(M)
    tons = [AMP * np.sin(2 * np.pi * f[s] * t) * rampe for s in sym]
    return np.concatenate([preambule(), np.zeros(int(GAP_T * FS))] + tons), sym


# ---------- récepteur ----------
def trouver_debut(signal):
    pre = preambule()
    c = np.abs(correlate(signal, pre, mode="valid", method="fft"))
    return int(np.argmax(c)) + len(pre) + int(GAP_T * FS)


def detecter_symboles(signal, M, debut, nb=None):
    n = int(TS * FS)
    fen = np.hanning(n)
    f_fft = np.fft.rfftfreq(n, 1 / FS)
    idx = np.array([np.argmin(np.abs(f_fft - f)) for f in frequences(M)])
    symboles = []
    i = debut
    while i + n <= len(signal) and (nb is None or len(symboles) < nb):
        spectre = np.abs(np.fft.rfft(signal[i:i + n] * fen))
        symboles.append(int(np.argmax(spectre[idx])))
        i += n
    return np.array(symboles)


def recevoir(signal, M):
    debut = trouver_debut(signal)
    k = int(np.log2(M))
    # 1) lire l'octet de longueur
    nb_entete = int(np.ceil(8 / k))
    entete = detecter_symboles(signal, M, debut, nb_entete)
    longueur = int(np.packbits(symboles_vers_bits(entete, M)[:8])[0])
    # 2) lire le message complet
    nb = int(np.ceil(8 * (1 + longueur) / k))
    sym = detecter_symboles(signal, M, debut, nb)
    return bits_vers_texte(symboles_vers_bits(sym, M)), sym


# ---------- synchronisation de secours (ajoutée après les premiers tests réels) ----------

def _nettete(x, M, d, nb):
    """Somme, sur nb tranches, de la part d'énergie de la raie la plus forte (1 = note parfaitement nette)."""
    n = int(TS * FS); fen = np.hanning(n); f = np.fft.rfftfreq(n, 1 / FS)
    idx = [int(np.argmin(np.abs(f - fr))) for fr in frequences(M)]
    s = 0.0
    for k in range(nb):
        seg = x[d + k * n: d + (k + 1) * n]
        if len(seg) < n:
            break
        sp = np.abs(np.fft.rfft(seg * fen))[idx]
        s += sp.max() / (sp.sum() + 1e-12)
    return s

def _bornes_son(x):
    """Début et fin de la zone où il y a du son (énergie par tranches de 10 ms)."""
    pas = int(0.01 * FS)
    e = np.array([np.sqrt(np.mean(x[i:i + pas] ** 2)) for i in range(0, len(x) - pas, pas)])
    bruit, fort = np.percentile(e, 10), np.percentile(e, 99)
    actif = np.where(e > bruit + 0.2 * (fort - bruit))[0]
    if len(actif) == 0:
        return None
    i0, i1 = actif[0] * pas, (actif[-1] + 1) * pas          # début et fin du son
    return i0, i1


def _fin_prevue(x, M, d):
    """Où le message devrait finir si il commence en d (lu dans l'octet de longueur)."""
    n = int(TS * FS); k_bits = int(np.log2(M)); nb_entete = int(np.ceil(8 / k_bits))
    ent = detecter_symboles(x, M, d, nb_entete)
    if len(ent) < nb_entete:
        return None
    L = int(np.packbits(symboles_vers_bits(ent, M)[:8])[0])
    return d + int(np.ceil(8 * (1 + L) / k_bits)) * n


def synchro_par_notes(x, M):
    """Synchronisation SANS préambule (si le micro a raté le chirp) :
    arrivée du son, calage fin sur la netteté des notes, puis choix du bon
    départ grâce à l'octet de longueur, qui doit annoncer la fin réelle du son."""
    n = int(TS * FS)
    bornes = _bornes_son(x)
    if bornes is None:
        return None
    i0, i1 = bornes
    # 1) calage fin (à moins d'une note près) sur les premières notes
    fin = max(range(i0, i0 + n, int(0.002 * FS)), key=lambda d: _nettete(x, M, d, 6))
    # 2) choix du bon numéro de note grâce à l'octet de longueur et à la fin du son
    nb_entete = int(np.ceil(8 / int(np.log2(M))))
    meilleur, ecart_min = None, None
    for k in range(-5, 3):
        d = fin + k * n
        if d < 0:
            continue
        if _nettete(x, M, d, nb_entete) < 0.5 * nb_entete:
            continue  # pas de vraies notes à cet endroit (silence ou bruit)
        fp = _fin_prevue(x, M, d)
        if fp is None:
            continue
        ecart = abs(fp - i1)
        if ecart_min is None or ecart < ecart_min:
            meilleur, ecart_min = d, ecart
    return meilleur


def trouver_debut_robuste(x, M):
    """Choisit le début du message. Renvoie (début, méthode).
    1) le chirp, si l'octet de longueur lu à cet endroit annonce bien la fin réelle du son ;
    2) sinon la synchro par les notes (micro qui a coupé le chirp), si elle est cohérente ;
    3) si aucune n'est confirmée (octet de longueur abîmé), celle où les notes sont les plus nettes."""
    n = int(TS * FS)
    bornes = _bornes_son(x)
    d_c = trouver_debut(x)
    if bornes is None:
        return d_c, "chirp"

    def coherent(d):
        fp = _fin_prevue(x, M, d)
        return fp is not None and abs(fp - bornes[1]) < n

    if coherent(d_c):
        return d_c, "chirp"
    d_n = synchro_par_notes(x, M)
    if d_n is not None and coherent(d_n):
        return d_n, "notes"
    if d_n is not None and _nettete(x, M, d_n, 10) > _nettete(x, M, d_c, 10) + 0.5:
        return d_n, "notes (non confirmé)"
    return d_c, "chirp (non confirmé)"
