# R packages for the native estimators. Install into the active R library.
options(repos = c(CRAN = "https://cloud.r-project.org"))
required <- c(pscl = "1.5.9", wnominate = "1.5", emIRT = "0.0.15")
for (package in c("jsonlite", "digest", "remotes")) {
  if (!requireNamespace(package, quietly = TRUE)) install.packages(package)
}
for (package in names(required)) {
  version <- required[[package]]
  if (!requireNamespace(package, quietly = TRUE) ||
      as.character(packageVersion(package)) != version) {
    remotes::install_version(package, version = version, upgrade = "never")
  }
  stopifnot(as.character(packageVersion(package)) == version)
}
