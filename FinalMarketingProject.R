
library(tidyverse)
library(reshape)
library(caret)
library(MCMCpack)
library("randomForest")
library("ROCit")
library("ROCR")
library(Boruta)
#Read train dataset
train = read.csv("train.csv", header=T, sep =";")
test = read.csv("test.csv", header=T, sep =";")
train$job = as.factor(train$job)
train$marital = as.factor(train$marital)
train$education = as.factor(train$education)
train$default = as.factor(train$default)
train$housing = as.factor(train$housing)
train$loan = as.factor(train$loan)
train$contact = as.factor(train$contact)
train$month = as.factor(train$month)
train$poutcome = as.factor(train$poutcome)

test$job = as.factor(test$job)
test$marital = as.factor(test$marital)
test$education = as.factor(test$education)
test$default = as.factor(test$default)
test$housing = as.factor(test$housing)
test$loan = as.factor(test$loan)
test$contact = as.factor(test$contact)
test$month = as.factor(test$month)
test$poutcome = as.factor(test$poutcome)

head(train)
train <- train %>%
  mutate(
    age_group = dplyr::case_when(
      age <= 20            ~ "0-20",
      age > 20 & age <= 30 ~ "20-30",
      age > 30 & age <= 40 ~ "30-40",
      age > 40 & age <= 50 ~ "40-50",
      age > 50 & age <= 60 ~ "50-60",
      age > 60 & age <= 70 ~ "60-70",
      age > 70             ~ "Above 70"
    ),
    # Convert to factor
    age_group = factor(age_group, 
                       level = c("0-20", "20-30","30-40","40-50","50-60","60-70", "Above 70") )
  )
ggplot(train, aes(y)) + 
  geom_bar(aes(y = (..count..)/sum(..count..)),color = "black",fill = "purple") + geom_text(aes( label = scales::percent( (..count..)/sum(..count..)),
                                                                 y=  (..count..)/sum(..count..) ), stat= "count", vjust = -.2) +
  scale_y_continuous(labels=scales::percent) +
  labs(title="Y Histogram", y="Percentage",x = "Y")+ theme(text = element_text(size=15), plot.title = element_text(hjust = 0.5))


train_dep = subset(train, y == 'yes')
train_no_dep = subset(train, y == 'no')
train_temp = subset(train, job == 'housemaid')
summary(train_temp)

fact <- c(2:5,7:9,11,16,17)
num <- c(1,6,10,12:15)
for (i in fact)
{
  print(names(train)[i])
  print("Train Data(%):")
  print(round(prop.table(table(train[,i]))*100,1)) 
 
}

for (i in num)
{
  print(names(train)[i])
  print("Train Data(%):")   
  print(summary(train[,i]))
 
}

any(is.null(train))

#Age histogram
options(repr.plot.width = 12, repr.plot.height = 8)
age_hist = ggplot(train, aes(age))
age_hist + geom_histogram(binwidth = 5, color = "black",fill = "blue") + labs(title="Age Histogram", 
                                                                              y="Count",
                                                                              x = "Age")+ theme(text = element_text(size=15), plot.title = element_text(hjust = 0.5))

#Job
num_job = train %>%
  group_by(job)  %>%  
  summarize(lenJob = length(job)) 

ggplot(num_job, aes(x=job, y=lenJob, fill=job)) +
  geom_bar(stat="identity", position='dodge')+ labs(title="Job Title Frequency", 
                                  y="Count",
                                  x = "Job",
                                  fill="Job Code")+
          theme(text = element_text(size=15), axis.text.x=element_text(angle = 90, vjust = 0.5, hjust=1,size=10), plot.title = element_text(hjust = 0.5, size=15))

#Education
num_education_dep = train_dep %>%
  group_by(education)  %>%  
  summarize(yes = length(education)) 

num_education_nodep = train_no_dep %>%
  group_by(education)  %>%  
  summarize(no = length(job)) 

num_education = merge(num_education_dep, num_education_nodep, by="education")
num_education = as.data.frame(num_education)
num_education = melt(num_education, id.vars = 'education')


ggplot(num_education, aes(x=education, y=value, fill=variable )) + 
  geom_bar(stat='identity', position='dodge') +
  labs(title="Education Frequency by Term Deposit", 
       y="Count",
       x = "Education Level",
       fill="Term Deposit")+
  theme(plot.title = element_text(hjust = 0.5))

balance_age = train %>%
  group_by(age_group)  %>%  
  summarize(averageBalance = mean(balance)) 

ggplot(balance_age, aes(x=age_group, y=averageBalance, fill=age_group)) +
  geom_bar(stat="identity", position='dodge')+ labs(title="Average Balance by Age", 
                                  y="Average Balance",
                                  x = "Age Group",
                                  fill="Age Group")+
  theme( plot.title = element_text(hjust = 0.5))



#Modeling
train <- train %>%
  mutate(y_int = ifelse(y == "no",0,1))

test <- test %>%
  mutate(y_int = ifelse(y == "no",0,1))

train.glm = 
  glm(y_int~age+job+marital+education+default+balance+housing+loan+contact+day
      +month+duration+campaign+pdays+previous+poutcome, data=train, family=binomial(link="logit"))

sink("lm.txt")
print(summary.glm(train.glm))
sink()
test$model_prob_lr <- predict(train.glm, test, type = "response")
test = test  %>% mutate(model_pred = 1*(model_prob_lr > .5) + 0)
test_lr <- test %>% mutate(accurate = 1*(model_pred == y_int))
lrl_accuracy = sum(test_lr$accurate)/nrow(test_lr)
confusionMatrix(as.factor(test_lr$model_pred), as.factor(test_lr$y_int),mode = "everything", positive="1")

threshold = .01
max_f1 = 0
best_threshold = 0
while (threshold < .99) {
  test = test  %>% mutate(model_pred = 1*(model_prob_lr > threshold) + 0)
  test_lr <- test %>% mutate(accurate = 1*(model_pred == y_int))
  lrl_accuracy = sum(test_lr$accurate)/nrow(test_lr)
  cm = confusionMatrix(as.factor(test_lr$model_pred), as.factor(test_lr$y_int),mode = "everything", positive="1")
  f1 = cm$byClass[7]
  if (f1 > max_f1){
     max_f1 = f1
     best_threshold = threshold
  }
  
  threshold = threshold + .01
  
}

class = train.glm$y
score = qlogis(train.glm$fitted.values)
rocit_emp = rocit(score, class, method = "emp")
plot(rocit_emp)
print(rocit_emp)


train.glm = 
  glm(y_int~age+job+marital+education+default+balance+housing+loan+contact+day
      +month+duration+campaign+pdays+previous+poutcome, data=train, family=binomial(link="logit"))
summary.glm(train.glm)
test$model_prob_lr <- predict(train.glm, test, type = "response")
test = test  %>% mutate(model_pred = 1*(model_prob_lr > best_threshold) + 0)
test_lr <- test %>% mutate(accurate = 1*(model_pred == y_int))
lrl_accuracy = sum(test_lr$accurate)/nrow(test_lr)
confusionMatrix(as.factor(test_lr$model_pred), as.factor(test_lr$y_int),mode = "everything", positive="1")


train.logit = MCMClogit(y_int~age+job+marital+education+default+balance+housing+loan+contact+day
             +month+duration+campaign+pdays+previous+poutcome, data=train, burin=1000, mcmc=6000,  thin=6)

sink("lm2.txt")
print(summary(train.logit))
sink()

mod.mcmcprobit =
  MCMClogit(y_int~age+job+marital+education+default+balance+housing+loan+contact+day
             +month+duration+campaign+pdays+previous+poutcome,
             data=train, burin=1000, mcmc=6000,  thin=6)
summary(mod.mcmcprobit)
