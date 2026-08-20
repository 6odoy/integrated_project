"""Cuadro 3 - Linea base en el conjunto de prueba local.

Cuatro modelos de dificultad creciente (seccion 4.2), misma particion,
misma semilla (42), umbral fijado en validacion, IC bootstrap al 95 %.

Uso:
    python3 src/baseline.py --manifest data/manifest.csv --images-root data/images
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_manifest, split_por_paciente, caracteristicas, SEED
from metricas import umbral_en_validacion, metricas, bootstrap_ic, ESPEC_MIN


# ---------------------------------------------------------------- modelos
def logreg_pca(Xtr, ytr, Xva, yva):
    """Estandarizacion -> PCA -> regresion logistica. Grid en validacion."""
    mejor = (-1, None, None)
    for n_comp in (16, 32, 48):
        for C in (0.01, 0.1, 1.0, 10.0):
            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("pca", PCA(n_components=min(n_comp, Xtr.shape[1], len(Xtr)), random_state=SEED)),
                ("clf", LogisticRegression(C=C, max_iter=2000,
                                           class_weight="balanced", random_state=SEED)),
            ]).fit(Xtr, ytr)
            auc = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
            if auc > mejor[0]:
                mejor = (auc, pipe, {"pca_n": n_comp, "C": C})
    return mejor[1], mejor[2]


def random_forest(Xtr, ytr, Xva, yva):
    mejor = (-1, None, None)
    for n_est in (300, 600):
        for depth in (None, 10, 20):
            for leaf in (1, 5):
                rf = RandomForestClassifier(
                    n_estimators=n_est, max_depth=depth, min_samples_leaf=leaf,
                    class_weight="balanced", n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
                auc = roc_auc_score(yva, rf.predict_proba(Xva)[:, 1])
                if auc > mejor[0]:
                    mejor = (auc, rf, {"n_estimators": n_est, "max_depth": depth,
                                       "min_samples_leaf": leaf})
    return mejor[1], mejor[2]


def mlp_64(Xtr, ytr, Xva, yva):
    """MLP de una capa oculta de 64 unidades, BCE ponderada por clase (torch).

    Se usa torch y no sklearn porque el documento exige PONDERACION DE LA PERDIDA
    y no remuestreo (seccion 3.3); MLPClassifier no acepta pesos de clase.
    """
    import torch, torch.nn as nn
    torch.manual_seed(SEED)
    sc = StandardScaler().fit(Xtr)
    xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    xv = torch.tensor(sc.transform(Xva), dtype=torch.float32)

    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    mejor_auc, mejores_pesos, paciencia = -1, None, 0
    for epoca in range(3000):
        net.train(); opt.zero_grad()
        lossf(net(xt), yt).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            auc = roc_auc_score(yva, torch.sigmoid(net(xv)).numpy().ravel())
        if auc > mejor_auc:                       # parada temprana por AUC de validacion
            mejor_auc, paciencia = auc, 0
            mejores_pesos = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            paciencia += 1
            if paciencia >= 30:
                break
    net.load_state_dict(mejores_pesos)

    def puntuar(X):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(sc.transform(X), dtype=torch.float32))).numpy().ravel()
    return puntuar, {"hidden": 64, "epocas_utiles": epoca + 1, "pos_weight": float(pos_w)}


# ---------------------------------------------------------------- tabla
def fila(nombre, y_te, s_te, pac_te, umbral, n_boot):
    m = metricas(y_te, s_te, umbral)
    ic, desc = bootstrap_ic(y_te, s_te, pac_te, umbral, n_boot=n_boot)
    f = {"modelo": nombre, "umbral": umbral, "boot_descartados": desc}
    for k, v in m.items():
        f[k] = v
        f[f"{k}_ic"] = ic.get(k, (np.nan, np.nan))
    return f


def fmt(v, ic):
    if isinstance(v, float) and np.isnan(v):
        return "--"
    if ic is None or (isinstance(ic[0], float) and np.isnan(ic[0])):
        return f"{v:.3f}".replace(".", ",")
    return f"{v:.3f} [{ic[0]:.3f}; {ic[1]:.3f}]".replace(".", ",")


def a_latex(filas, prevalencia):
    cols = [("sens", "Sens. @esp$\\geq$0,80"), ("espec", "Especif."),
            ("auc_roc", "AUC-ROC"), ("auprc", "AUPRC"), ("f1", "F1")]
    out = [
        "% Cuadro 3 - generado por src/baseline.py",
        "\\begin{tabular}{lccccc}", "\\hline",
        "\\textbf{Modelo} & " + " & ".join(f"\\textbf{{{t}}}" for _, t in cols) + " \\\\",
        "\\hline",
    ]
    for f in filas:
        out.append(f["modelo"] + " & " + " & ".join(fmt(f[k], f[f"{k}_ic"]) for k, _ in cols) + " \\\\")
    out.append("\\hline")
    out.append("\\multicolumn{6}{l}{\\footnotesize Prevalencia en prueba: "
               + f"{prevalencia:.3f}".replace(".", ",")
               + ". El AUPRC de un modelo sin senal tiende a la prevalencia.} \\\\")
    out += ["\\hline", "\\end{tabular}"]
    return "\n".join(out)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images-root", default="")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    print("\n[1/5] Manifiesto")
    df, incert = load_manifest(a.manifest)
    incert.to_csv(os.path.join(a.out, "conjunto_incertidumbre_fase3.csv"), index=False)

    print("[2/5] Particion por paciente (70/15/15, semilla 42)")
    df = split_por_paciente(df, SEED)
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        print(f"  {s:5} imagenes={len(d):6}  pacientes={d.patient_id.nunique():5}  prevalencia={d.label.mean():.4f}")
    df.to_csv(os.path.join(a.out, "particion.csv"), index=False)

    print("[3/5] Caracteristicas (histograma de intensidad, 128x128)")
    sub = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    X = {s: caracteristicas(sub[s], a.images_root) for s in sub}
    y = {s: sub[s]["label"].to_numpy() for s in sub}
    pac_te = sub["test"]["patient_id"].to_numpy()

    print("[4/5] Entrenamiento (hiperparametros elegidos en VALIDACION)")
    filas, hp = [], {}
    prev = float(y["test"].mean())

    # (i) trivial: predice SIEMPRE la clase mayoritaria del ENTRENAMIENTO.
    # No se cablea: en Kermany la mayoritaria es neumonia, en BRAX es "sin hallazgos".
    mayoritaria = int(round(float(y["train"].mean())))
    s_triv = np.full(len(y["test"]), float(mayoritaria))
    filas.append(fila(f"Trivial (clase mayoritaria = {mayoritaria})", y["test"],
                      s_triv, pac_te, 0.5, a.n_boot))
    print(f"  clase mayoritaria en entrenamiento: {mayoritaria} "
          f"(prevalencia train = {y['train'].mean():.4f})")

    for nombre, ajusta in (("Regresion logistica + PCA", logreg_pca),
                           ("Random Forest", random_forest),
                           ("MLP minimo (64 unidades)", mlp_64)):
        mod, params = ajusta(X["train"], y["train"], X["val"], y["val"])
        pred = mod if callable(mod) and not hasattr(mod, "predict_proba") \
            else (lambda Z, m=mod: m.predict_proba(Z)[:, 1])
        s_val, s_te = pred(X["val"]), pred(X["test"])
        u = umbral_en_validacion(y["val"], s_val, ESPEC_MIN)
        filas.append(fila(nombre, y["test"], s_te, pac_te, u, a.n_boot))
        hp[nombre] = params
        print(f"  {nombre:28} umbral(val)={u:.4f}  hp={params}")

    print("[5/5] Salidas")
    tex = a_latex(filas, prev)
    open(os.path.join(a.out, "cuadro3.tex"), "w").write(tex)
    pd.DataFrame(filas).to_csv(os.path.join(a.out, "cuadro3.csv"), index=False)
    json.dump({"semilla": SEED, "espec_min": ESPEC_MIN, "n_boot": a.n_boot,
               "prevalencia_test": prev, "hiperparametros": hp},
              open(os.path.join(a.out, "config.json"), "w"), indent=2, default=str)
    print("\n" + tex)
    print(f"\nListo en {time.time()-t0:.1f}s -> {a.out}/")


if __name__ == "__main__":
    main()
