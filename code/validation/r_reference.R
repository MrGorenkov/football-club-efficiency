# Reference estimates from independent R implementations: Benchmarking (Bogetoft & Otto) for DEA and
# frontier (Coelli & Henningsen) for SFA. Writes CSV files next to this script.
suppressMessages({library(Benchmarking); library(frontier)})
out <- dirname(normalizePath(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
data(charnes1981)
X <- as.matrix(charnes1981[, c("x1", "x2", "x3", "x4", "x5")])
Y <- as.matrix(charnes1981[, c("y1", "y2", "y3")])
write.csv(charnes1981, file.path(out, "charnes1981.csv"), row.names = FALSE)
# Benchmarking reports the Farrell output measure (>= 1) for ORIENTATION = "out"
d <- data.frame(vrs = eff(dea(X, Y, RTS = "vrs", ORIENTATION = "out")),
                crs = eff(dea(X, Y, RTS = "crs", ORIENTATION = "out")))
write.csv(d, file.path(out, "r_dea_charnes1981.csv"), row.names = FALSE)

data(front41Data)
m1 <- sfa(log(output) ~ log(capital) + log(labour), data = front41Data)
write.csv(front41Data, file.path(out, "front41Data.csv"), row.names = FALSE)
write.csv(data.frame(name = names(coef(m1)), est = coef(m1), se = sqrt(diag(vcov(m1))), loglik = logLik(m1)[1]),
          file.path(out, "r_sfa_front41.csv"), row.names = FALSE)
write.csv(data.frame(te = efficiencies(m1)[, 1]), file.path(out, "r_sfa_front41_te.csv"), row.names = FALSE)

data(riceProdPhil)
m2 <- sfa(log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) | EDYRS + BANRAT, data = riceProdPhil)
write.csv(riceProdPhil, file.path(out, "riceProdPhil.csv"), row.names = FALSE)
write.csv(data.frame(name = names(coef(m2)), est = coef(m2), se = sqrt(diag(vcov(m2))), loglik = logLik(m2)[1]),
          file.path(out, "r_sfa_rice_bc95.csv"), row.names = FALSE)
write.csv(data.frame(te = efficiencies(m2)[, 1]), file.path(out, "r_sfa_rice_te.csv"), row.names = FALSE)
cat("done\n")
