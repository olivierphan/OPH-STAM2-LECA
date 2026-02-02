library(terra)
library(fs)

# Fonction pour fusionner les rasters
merge_rasters <- function(raster_list) {
  if (length(raster_list) == 0) {
    stop("Aucun fichier .asc trouvé.")
  }
  if (length(raster_list) == 1) {
    return(raster_list[[1]])
  }
  merged <- raster_list[[1]]
  for (i in 2:length(raster_list)) {
    merged <- terra::merge(merged, raster_list[[i]])
  }
  return(merged)
}

# Chemin du dossier principal contenant les sous-dossiers avec les fichiers .asc
main_folder <- "C:/Users/phano/Documents/_MoBI 2024/LECA2025/Topo/BDALTI"

# Trouver tous les fichiers .asc dans les sous-dossiers
asc_files <- dir_ls(path = main_folder, recurse = TRUE, glob = "*.asc")

# Lire tous les fichiers .asc
cat("Chargement des fichiers .asc...\n")
raster_list <- lapply(asc_files, function(f) {
  cat("  Chargement de", basename(f), "\n")
  terra::rast(f)
})

# Fusionner tous les rasters
cat("Fusion des rasters...\n")
merged_raster <- merge_rasters(raster_list)

# Définir le système de coordonnées si nécessaire
crs(merged_raster) <- "EPSG:2154"

# Chemin de sortie pour le fichier .tif
output_file <- file.path(main_folder, "BDALTIV2-0-25M_LAM93-IGN69.tif")

# Écrire le raster fusionné en format .tif
cat("Sauvegarde du raster fusionné...\n")
terra::writeRaster(merged_raster, output_file, overwrite = TRUE)

cat("Fusion terminée. Fichier sauvegardé sous:", output_file, "\n")
