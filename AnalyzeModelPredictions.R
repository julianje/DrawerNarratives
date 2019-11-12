library(tidyverse)

setwd("~/Documents/Projects/Models/DrawerNarratives/")

Predictions <- read_csv("DrawerPredictions.csv")

# Make things more legible:
Predictions$Actions <- str_remove_all(Predictions$Actions,"m")
Predictions$Knowledge <- factor(Predictions$Knowledge)
#levels(Predictions$Knowledge) = c("Approx 0,0","Approx 0,1","Approx 0,2","Approx 0,3","Approx 1,0",
#                                  "Approx 1,1","Approx 1,2","Approx 1,3","Approx 2,0","Approx 2,1",
#                                  "Approx 2,2","Approx 2,3","Approx 3,0","Approx 3,1","Approx 3,2",
#                                  "Approx 3,3","Color(0)","Color(1)","Color(2)","First column",
#                                  "Second column","Third column","Fourth column","Ignorant","")

BuildActionPlot <- function(actions){
  actions <- data.frame(str_split(str_split(actions,"_")[[1]],"-"))
  names(actions)=1:length(names(actions))
  actions$Coord = c("x","y")
  p <- actions %>% gather(Time,pos,1:4) %>%
    mutate(pos=as.integer(pos),Time=as.integer(Time)) %>%
    spread(Coord,pos) %>%
    ggplot(aes(x,y,label=Time))+
    scale_x_continuous("",limits=c(-0.5,3.5), breaks=c(0,1,2,3))+
    scale_y_continuous("",limits=c(-0.5,3.5), breaks=c(0,1,2,3))+
    geom_text()+theme_classic()+
    geom_hline(aes(yintercept=c(0.5,1.5,2.5,3.5)))+
    geom_vline(aes(xintercept=c(0.5,1.5,2.5,3.5)))+
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())
  return(p)
}

VisualizeMarginals <- function(TrialName){
  Subset <- Predictions %>% filter(TrialId==TrialName)
  SubA <- Subset %>% group_by(Actions) %>%
    summarise(Probability=sum(Probability)) %>%
    rename(Hypothesis=Actions) %>% mutate(Type="Actions")
  SubB <- Subset %>% group_by(Knowledge) %>%
    summarise(Probability = sum(Probability)) %>%
    rename(Hypothesis=Knowledge) %>% mutate(Type="Knowledge")
  bind_rows(SubA,SubB) %>% filter(Probability>0) %>%
    ggplot(aes(Hypothesis,Probability))+geom_bar(stat="identity")+theme_classic()+
    facet_wrap(~Type,scales="free")+coord_flip()
}

VisualizeJoint <- function(TrialName){
  Predictions %>% filter(TrialId==TrialName) %>%
    filter(Probability > 0) %>%
    ggplot(aes(Knowledge,Actions,fill=Probability))+geom_tile()+
    facet_wrap(~TrialId,scales="free")+theme_classic()+
    scale_fill_gradient(low="#ffffff",high="red")
}

Trials <- unique(Predictions$TrialId)

VisualizeMarginals(Trials[7])

VisualizeJoint(Trials[10])

# Get best hypotheses:
Predictions %>% group_by(TrialId) %>%
  top_n(2,Probability) %>%
  ggplot(aes(Knowledge,Actions,fill=Probability))+
  geom_tile()+facet_wrap(~TrialId,scales="free")+
  scale_fill_gradient(low="#ffffff",high="red")+
  theme_classic()
