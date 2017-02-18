library(jsonlite)
library(lattice)

test = stream_in(file(in_file_name))

case_cols = data.frame(
  red = c(1, 0, 0, .5),
  green = c(0, 0, 1, .5),
  blue = c(0, 1, 0, .5)
)

bw_cols = data.frame(
  red = c(0),
  green = c(0),
  blue = c(0)
)

last.cloud = test[test$timestep == max(test$timestep),]
last.cloud.vot = data.frame(Agent = c(), Lemma = c(), Case = c(), VOT = c())
for(i in 1:nrow(last.cloud)) {
  for(j in 1:nrow(last.cloud[i,]$exemplars[[1]])) {
    last.cloud.vot = rbind(last.cloud.vot, list(
      Agent = last.cloud[i,]$agent,
      Lemma = last.cloud[i,]$exemplars[[1]][j,]$lemma,
      Case = last.cloud[i,]$exemplars[[1]][j,]$case,
      VOT1 = last.cloud[i,]$exemplars[[1]][j,]$segments[[1]]$features$vot[1],
      VOT3 = last.cloud[i,]$exemplars[[1]][j,]$segments[[1]]$features$vot[3]))
    last.cloud.vot$Case = as.character(last.cloud.vot$Case)
  }
}

vot.tracker = expand.grid(Timestep = seq(0, max(test$timestep)), Agent = names(table(test$agent)), Lemma = lemmas, Case = c("abs", "erg"), Position = current_positions)
if(disperse) {
  vot.tracker = vot.tracker[!(vot.tracker$Lemma < 3 & vot.tracker$Position == 1),]
}
for(i in 1:nrow(vot.tracker)) {
  cloud = test[test$timestep == vot.tracker$Timestep[i] & test$agent == vot.tracker$Agent[i],]$exemplars[[1]]
  cloud = cloud[cloud$lemma == vot.tracker$Lemma[i] & cloud$case == vot.tracker$Case[i],]
  vot.cloud = cloud$segments
  vots = unlist(lapply(vot.cloud, function(x) x$features$vot[vot.tracker$Position[i]]))
  qs = quantile(vots, probs = c(0, .25, .5, .75, 1))
  vot.tracker$VOT_MIN[i] = qs[1]
  vot.tracker$VOT_Q1[i] = qs[2]
  vot.tracker$VOT_MED[i] = qs[3]
  vot.tracker$VOT_Q3[i] = qs[4]
  vot.tracker$VOT_MAX[i] = qs[5]
}

plot.vot.trackers <- function(vot_data, agent) {
  cases = names(table(vot_data$Case))
  positions = current_positions
  the_plot = xyplot(VOT_MED ~ Timestep | paste(Case, ", C", as.character(Position), sep = "") + paste("Lemma ", as.character(Lemma)),
    data = vot_data[vot_data$Agent == agent,],
    xlim = c(-150, max(vot_data$Timestep) + 150),
    xlab = "Iteration",
    ylim = c(-10, 110),
    ylab = "VOT",
    as.table = T,
    strip = function(which.given, which.panel, var.name, factor.levels, ...) {
      if(which.given == 1) {
        current_lemma = which.panel[2]
        panel.rect(0, 0, 1, 1, col = "grey", border = "transparent")
        if(which.panel[which.given] %% length(positions) == 1) {
          panel.lines(x = c(0, 0, 1), y = c(1, 0, 0), col = "black")
        }
        else {
          case_strings = factor.levels
          current_case = substr(case_strings[which.panel[which.given]], 1, 3)
          current_case_label = ifelse(current_case == "abs", "Absolutive", "Ergative")
          last_cloud_data = last.cloud.vot[last.cloud.vot$Agent == agent & last.cloud.vot$Lemma == current_lemma & last.cloud.vot$Case == current_case,]
          S1 = ifelse(median(last_cloud_data$VOT1) < 50, "b", "p")
          if(is.na(S1)) { S1 = "i" }
          S2 = "i"
          S3 = ifelse(median(last_cloud_data$VOT3) < 50, "b", "p")
          if(is.na(S3)) { S3 = "i" }
          current_label = paste(current_case_label, ": [", S1, S2, S3, sep = "")
          if(current_case == "erg") {
              current_label = paste(current_label, "i", sep = "")
          }
          current_label = paste(current_label, "]", sep = "")
          my.x.lines = c(0, 1, 1)
          my.y.lines = c(0, 0, 1)
          if(length(positions) == 1) {
            my.x.lines = c(0, 0, 1, 1)
            my.y.lines = c(1, 0, 0, 1)
          }
          panel.lines(x = my.x.lines, y = my.y.lines, col = "black")
          my.y.offset = 0
          if(length(positions) == 1) { my.y.offset = .5 }
          panel.text(my.y.offset, .25, labels = current_label)
        }
      }
      if(which.given == 2) {
        panel.rect(0, .5, 1, 1, col = "black")
        if(which.panel[1] == length(cases) * length(positions)) {
          panel.text(-1 * (length(positions) - 1), .75, col = "white", labels = factor.levels[which.panel[which.given]])
        }
      }
    },
    panel = function(x, y, subscripts) {
      temp_data = vot_data[vot_data$Agent == agent,]
      temp_lemma = temp_data[subscripts[1],"Lemma"]
      temp_case = temp_data[subscripts[1],"Case"]
      temp_position = temp_data[subscripts[1],"Position"]
      temp_data = temp_data[temp_data$Lemma == temp_lemma & temp_data$Case == temp_case & temp_data$Position == temp_position,]
      panel.lines(x = temp_data$Timestep, y = temp_data$VOT_MED, col = "black", lwd = 2.5)
      xx = c(temp_data$Timestep, rev(temp_data$Timestep))
      yy = c(temp_data$VOT_Q1, rev(temp_data$VOT_Q3))
      panel.polygon(xx, yy, col = do.call(rgb, c(bw_cols[1,], .4)), border = NA)
      yy = c(temp_data$VOT_MIN, rev(temp_data$VOT_MAX))
      panel.polygon(xx, yy, col = do.call(rgb, c(bw_cols[1,], .15)), border = NA)
    })
  return(the_plot)
}
