# Installs the external packages
devtools::install_github("curso-r/treesnip")
# look up latest release and install.
location <- "https://github.com/catboost/catboost/releases/download/v0.23.2/catboost-R-Darwin-0.23.2.tgz"
devtools::install_url(location, INSTALL_opts = c("--no-multiarch"))
