library(terra)
library(readxl)
library(tidyverse)
library(writexl)

oc_meta_path <- "data/orchamp/orchamp_meta.xlsx"
mnt_path <- "data/topo/BDALTI/BDALTIV2-0-25M_LAM93-IGN69.tif"
roads73_path <- "data/topo/BDTOPO/D073/ROUTE_NUMEROTEE_OU_NOMMEE.shp"
roads84_path <- "data/topo/BDTOPO/D084/ROUTE_NUMEROTEE_OU_NOMMEE.shp"


oc_meta <- read_excel(oc_meta_path)

### INITIATILISATION

roads73 <- vect(roads73_path)
roads84 <- vect(roads84_path)
roads <- rbind(roads73,roads84)
roads <- roads[roads$TYPE_ROUTE!="Route nommée",]


points <- vect(oc_meta, geom=c("longitude", "latitude"), crs="EPSG:4326") #en WGS84
points <- project(points, "EPSG:2154") #en Lambert 93

mnt <- rast(mnt_path)

### FONCTIONS

calculate_segments <- function(vect_road, vect_point,color) {
  line <- vect(rbind(geom(vect_road), geom(vect_point)), type="lines", crs="EPSG:2154")
  coords <- crds(line)
  n_segments <- ceiling(vect_road$distance/5) 
  points_segment <- matrix(NA, nrow = n_segments + 1, ncol = ncol(coords))
  colnames(points_segment) <- c("x","y")
  points_segment[1,] <- coords[1,]
  points_segment[n_segments + 1,] <- coords[nrow(coords),]
  distance_cumulee <- 0
  segment_actuel <- 2
  for (k in 2:nrow(coords)) {
    segment_distance <- sqrt(sum((coords[k,] - coords[k-1,])^2))
    while (distance_cumulee + segment_distance >= (segment_actuel - 1) * 5 && segment_actuel <= n_segments) {
      fraction <- ((segment_actuel - 1) * 5 - distance_cumulee) / segment_distance
      points_segment[segment_actuel,] <- coords[k-1,] + fraction * (coords[k,] - coords[k-1,])
      segment_actuel <- segment_actuel + 1
    }
    distance_cumulee <- distance_cumulee + segment_distance
  }
  
  lines(line, col = color,arrows = TRUE, lwd = 1)
  return(points_segment)
}

create_profile <- function(plot_name, segments, mnt) {
  vect_points <- vect(segments,  crs=crs(mnt))
  altitudes <- terra::extract(mnt, vect_points)[,2]
  if(is.na(altitudes[1])){
    j <- 1
    print(paste0(plot_name, " - Altitude initiale manquante"))
    while (is.na(altitudes[j])){
      j <- j+1
      if(j==nrow(segments)){
        print(paste0(plot_name," - Aucune donnée d'altitude trouvée"))
        break
      }
      
      if(!is.na(altitudes[j])){
        altitudes[1:j] <- altitudes[1+j]
        print(paste0(plot_name, " - Altitudes inititales manquantes consécutives : ",(j-1)))
        break
      }
    }
  }
  topographic_profile <- lapply(1:(nrow(segments)-1), function(i) {
    point1 <- segments[i,]
    point2 <- segments[i+1,]
    alti1 <- altitudes[i]
    
    if(is.na(altitudes[i+1])){
      j<-1
      print(paste0(plot_name, " - Altitude intermédiaire manquante"))
      while (is.na(altitudes[i+j])){
        j <- j+1
        if(j+1==nrow(segments)){
          print(paste0(plot_name," - Aucune donnée d'altitude trouvée"))
          break
        }
        if(!is.na(altitudes[i+j])){
          altitudes[i:i+j-1] <- altitudes[i] + (altitudes[i+j]-altitudes[i])/j
          print(paste0(placette, " - Altitudes intermédiaires manquantes consécutives : ",j))
          break
        } 
      } 
    }
    
    alti2 <- altitudes[i+1]
    dist_horiz <- terra::distance(vect(t(point1), crs=crs(mnt)), 
                                  vect(t(point2), crs=crs(mnt)))
    steep <- alti2 - alti1
    dist_ground <- sqrt(dist_horiz^2 + steep^2)
    list(
      start = list(x = point1[[1]], y = point1[[2]], altitude = altitudes[i]),
      end = list(x = point2[[1]], y = point2[[2]], altitude = altitudes[i+1]),
      gps_distance = dist_horiz,
      ground_distance = dist_ground,
      steep = steep
    )
  })
}

calculate_distances <- function(road,profile){
  aerial_distance <- 0
  progress <- 0
  dist_x <- 0
  dist_ground <-0
  leap_start <- 1
  leap_length <- 1
  while(dist_x<road$distance-5){
    progress <- progress +1
    starting_point <- vect(as.data.frame(profile[[leap_start]]$start),geom=c("x","y"),crs = "EPSG:2154")
    ending_point_temp <- vect(as.data.frame(profile[[leap_start+leap_length]]$start),geom=c("x","y"),crs = "EPSG:2154")
    future_point_temp <- vect(as.data.frame(profile[[leap_start+leap_length]]$end),geom=c("x","y"),crs = "EPSG:2154")
    leap_y <- ending_point_temp$altitude-starting_point$altitude
    leap_x <- distance(ending_point_temp,starting_point)[1]
    leap_x_next <-distance(future_point_temp,starting_point)[1]
    if (future_point_temp$altitude > starting_point$altitude + leap_x_next*leap_y/leap_x){
      leap_length <- leap_length + 1
    } else {
      aerial_distance <- aerial_distance + sqrt((leap_y^2+leap_x^2))
      leap_start <- leap_start + leap_length
      leap_length <- 1
    }
    dist_x <- dist_x + profile[[progress]]$gps_distance[1] 
    dist_ground <- dist_ground + profile[[progress]]$ground_distance[1]
    if (dist_x>road$distance-5){
      progress = progress+1
      aerial_distance <- aerial_distance + sqrt((leap_y^2+leap_x_next^2))
      dist_x <- dist_x + profile[[progress]]$gps_distance[1] 
      dist_ground <- dist_ground + profile[[progress]]$ground_distance[1]
    }
  }
  steeps <- sapply(profile, function(x) x$steep)
  cumulated_positive_steep <- sum(steeps[steeps > 0])
  cumulated_negative_steep <- sum(steeps[steeps < 0])
  return(list(
    gps_distance_tot = dist_x ,
    ground_distance_tot = dist_ground ,
    aerial_distance_tot = aerial_distance,
    cumulated_positive_steep = cumulated_positive_steep,
    cumulated_negative_steep = cumulated_negative_steep
    ))
}

##### Calcul principal

dist_topo <- function(i) {
  point <- points[i,]
  placette <- points[i,]$code.placette
  print(paste0(placette, " - Starting."))
  
  nearest_road_to_point <- nearest(point, roads)
  
  
  if(is.na(nearest_road_to_point$to_id)){
    nearest_road_to_point$to_id <- nearby(point,roads)[,2]
    nearest_road_name <- roads[nearby(point,roads)[,2]]$NUMERO
  } else {
    nearest_road_name <- roads[nearest_road_to_point$to_id,]$NUMERO
  }
  print(paste0(placette, " - Nearest road by GPS ", nearest_road_name, " : début de l'extraction des données."))
  print(paste0(placette, " - Nearest road by GPS ", nearest_road_name, " : point proche trouvé."))
  
  buffer <- buffer(point, width = nearest_road_to_point$distance+10000) #buffer = route proche +5km
  buffer_mnt_crs <- project(buffer, crs(mnt))
  mnt_crop <- crop(mnt, buffer_mnt_crs)
  mnt_L93 <- project(mnt_crop, "EPSG:2154")
  print(paste0(placette, " - Nearest road by GPS ", nearest_road_name, " : crop MNT OK."))
  
  plot(mnt_L93)
  points(point, col="red")
  lines(roads,col = "black", lwd = 1)
  
  segmented_line <- calculate_segments(nearest_road_to_point,point, color = 'blue')
  print(paste0(placette, " - Nearest road by GPS ", nearest_road_name, " : segmentation OK."))
  
  topographic_profile <- create_profile(placette,segmented_line,mnt_L93)
  
  road_to_point_distances <- calculate_distances(nearest_road_to_point, topographic_profile)
  
  print(paste0(placette, " - Nearest road by GPS ", nearest_road_name, " [GPS distance: ", road_to_point_distances$gps_distance_tot,"]"))
  print(paste0(placette, " - Nearest road by GPS ", nearest_road_name, " [Ground distance: ", road_to_point_distances$ground_distance_tot,"]"))
  print(paste0(placette, " - Nearest road by GPS ", nearest_road_name, " [aerial distance: ", road_to_point_distances$aerial_distance_tot,"]"))
  
  df <- data.frame(
    nearest_road = nearest_road_name,
    GPS_distance = road_to_point_distances$gps_distance_tot,
    ground_distance = road_to_point_distances$ground_distance_tot,
    aerial_distance = road_to_point_distances$aerial_distance_tot,
    cumulated_positive_steep = road_to_point_distances$cumulated_positive_steep,
    cumulated_negative_steep = road_to_point_distances$cumulated_negative_steep
  )
  print(paste0(placette, " - Route proche ", nearest_road_name, " : Done."))
  
  if (road_to_point_distances$aerial_distance_tot > road_to_point_distances$gps_distance_tot) {
    print(paste0(placette, " - Looking for closer roads. "))
    better_roads <- nearby(point, roads, distance = road_to_point_distances$aerial_distance_tot)
    for(candidates in better_roads[,"to_id"]){
      candidate <- nearest(point, roads[candidates,])
      if(is.na(candidate$to_id)){
        candidate$to_id <- 0
        candidate_name <- "Unnamed road"
      } else {
        candidate_name <- roads[candidates,]$NUMERO
      }
      if(candidate_name!=df[1,]$nearest_road){
        print(paste0(placette, " - New candidate:  ",candidate_name ))
        segmented_candidate <- calculate_segments(candidate,point,color='blue')
        topo_profile_candidate <- create_profile(placette,segmented_candidate,mnt_L93)
        candidate_distances <- calculate_distances(candidate,topo_profile_candidate)
        df_candidate <-data.frame(
          nearest_road = candidate_name,
          GPS_distance = candidate_distances$gps_distance_tot,
          ground_distance = candidate_distances$ground_distance_tot,
          aerial_distance = candidate_distances$aerial_distance_tot,
          cumulated_positive_steep = candidate_distances$cumulated_positive_steep,
          cumulated_negative_steep = candidate_distances$cumulated_negative_steep
        )
        df <- rbind(df, df_candidate)
      }
      print(paste0(placette, " - New candidate:  ",candidate_name, " done."))
    }
  }
  nearest_road_by_air <- roads[roads$NUMERO==df[which.min(df$aerial_distance),]$nearest_road,]
  print(paste0(placette, " - Nearest road by air: ", nearest_road_by_air$NUMERO))
  nearest_line <- vect(rbind(geom(nearest(point,nearest_road_by_air)), geom(point)), type="lines", crs="EPSG:2154")
  lines(nearest_line,col="red")
  print(paste0(placette, " - Nearest road by air ", nearest_road_by_air$NUMERO, " [GPS distance: ", df[which.min(df$aerial_distance),]$GPS_distance,"]"))
  print(paste0(placette, " - Nearest road by air ", nearest_road_by_air$NUMERO, " [Ground distance: ", df[which.min(df$aerial_distance),]$ground_distance,"]"))
  print(paste0(placette, " - Nearest road by air ", nearest_road_by_air$NUMERO, " [aerial distance: ", df[which.min(df$aerial_distance),]$aerial_distance,"]"))
  print(paste0(placette, " - Done."))
  
  return(df)
}

distances_df <- oc_meta %>%
  add_column(
    GPS_nearest_road=NA,
    GPS_GPS_distance=NA,
    GPS_ground_distance=NA,
    GPS_aerial_distance=NA,
    GPS_cumulated_positive_steep=NA,
    GPS_cumulated_negative_steep=NA,
    GRD_nearest_road=NA,
    GRD_GPS_distance=NA,
    GRD_ground_distance=NA,
    GRD_aerial_distance=NA,
    GRD_cumulated_positive_steep=NA,
    GRD_cumulated_negative_steep=NA,
    AIR_nearest_road=NA,
    AIR_GPS_distance=NA,
    AIR_ground_distance=NA,
    AIR_aerial_distance=NA,
    AIR_cumulated_positive_steep=NA,
    AIR_cumulated_negative_steep=NA,
    
  )
for (l in 1:nrow(points)){
  if(any(is.na(distances_df[l,]))){
    nearest_roads <- dist_topo(l)
    GPS_nearest <- nearest_roads[which.min(nearest_roads$GPS_distance),]
    distances_df[l,]$GPS_nearest_road <- GPS_nearest$nearest_road
    distances_df[l,]$GPS_GPS_distance <- GPS_nearest$GPS_distance
    distances_df[l,]$GPS_ground_distance <- GPS_nearest$ground_distance
    distances_df[l,]$GPS_aerial_distance <- GPS_nearest$aerial_distance
    distances_df[l,]$GPS_cumulated_positive_steep <- GPS_nearest$cumulated_positive_steep
    distances_df[l,]$GPS_cumulated_negative_steep <- GPS_nearest$cumulated_negative_steep
    
    GRD_nearest <- nearest_roads[which.min(nearest_roads$ground_distance),]
    distances_df[l,]$GRD_nearest_road <- GRD_nearest$nearest_road
    distances_df[l,]$GRD_GPS_distance <- GRD_nearest$GPS_distance
    distances_df[l,]$GRD_ground_distance <- GRD_nearest$ground_distance
    distances_df[l,]$GRD_aerial_distance <- GRD_nearest$aerial_distance
    distances_df[l,]$GRD_cumulated_positive_steep <- GRD_nearest$cumulated_positive_steep
    distances_df[l,]$GRD_cumulated_negative_steep <- GRD_nearest$cumulated_negative_steep
    
    AIR_nearest <- nearest_roads[which.min(nearest_roads$aerial_distance),]
    distances_df[l,]$AIR_nearest_road <- AIR_nearest$nearest_road
    distances_df[l,]$AIR_GPS_distance <- AIR_nearest$GPS_distance
    distances_df[l,]$AIR_ground_distance <- AIR_nearest$ground_distance
    distances_df[l,]$AIR_aerial_distance <- AIR_nearest$aerial_distance
    distances_df[l,]$AIR_cumulated_positive_steep <- AIR_nearest$cumulated_positive_steep
    distances_df[l,]$AIR_cumulated_negative_steep <- AIR_nearest$cumulated_negative_steep
  }
}

