library(tidyverse)

setwd("~/Documents/Projects/Models/DrawerNarratives/")

Predictions <- read_csv("DrawerPredictions.csv")

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
    filter(Probability != 0) %>%
    ggplot(aes(Knowledge,Actions,fill=Probability))+geom_tile()+
    facet_wrap(~TrialId,scales="free")+theme_classic()+
    scale_fill_gradient(low="#ffffff",high="red")
}

Trials <- unique(Predictions$TrialId)

VisualizeMarginals(Trials[1])
VisualizeJoint(Trials[7])
