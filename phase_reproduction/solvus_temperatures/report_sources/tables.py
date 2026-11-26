"""Format the coexistence and control rows recorded in results.json for XeLaTeX."""


def num(value, digits=2):
    return f"{value:.{digits}f}".replace(".", "{,}")


def contacts(samples):
    lines = [r"\begin{center}\footnotesize\setlength{\tabcolsep}{4.6pt}",
             r"\begin{tabular}{@{}rrlccr@{}}\toprule",
             r"\(N\)& \(W_{\max}\) & \(c/a\) & \(x_\alpha\), ат.\% Al & \(x_{\alpha_2}\), ат.\% Al & \(\beta\Delta F\)\\\midrule"]
    for record in samples:
        fit = record["phase_fit"]
        if fit is None:
            continue
        pairs = []
        for value, interval in zip(fit["Al_at_pct"], fit["Al_at_pct_conditional_95"]):
            label = num(value)
            if interval:
                label += r"\ ["+num(interval[0])+r";\,"+num(interval[1])+"]"
            pairs.append(r"\("+label+r"\)")
        barrier = fit["equal_height_barrier_kBT"]
        lines.append(" & ".join([str(record["N"]), r"\("+num(record["cap"])+r"\)",
                                 "своб." if record["axial_free"] else "фикс.", *pairs,
                                 r"\("+num(barrier,1)+r"\)" if barrier is not None else "---"]) + r"\\")
    lines.extend([r"\bottomrule\end{tabular}\end{center}"])
    return "\n".join(lines)
