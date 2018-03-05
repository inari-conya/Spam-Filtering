library(tm)
library(ggplot2)

setwd("F:\\ML\\ML_for_Hackers-master\\03-Classification") 

spam.path <- "data/spam"
spam2.path <- "data/spam_2"
easyham.path <- "data/easy_ham"
easyham2.path <- "data/easy_ham_2"
hardham.path <- "data/hard_ham"
hardham2.path <- "data/hard_ham_2"

get.msg <- function(path) {
  con <- file(path, open = "rt", encoding = "native.enc")
  text <- readLines(con)
  msg <- text[seq(which(text == "")[1]+1, length(text), 1)]
  close(con)
  return(paste(msg, collapse = "\n"))
}

spam.docs <- dir(spam.path)
spam.docs <- spam.docs[which(spam.docs != "cmds")]
all.spam <- sapply(spam.docs, function(p) get.msg(paste(spam.path, p, sep = "/")))
all.spam <- iconv(all.spam,"WINDOWS-1252","UTF-8")

get.tdm <- function(doc.vec) {
  doc.corpus <- Corpus(VectorSource(doc.vec))
  control <- list(stopwords = TRUE, removePunctuation = TRUE, 
                  removeNumbers = TRUE, minDocFreq = 2)
  doc.dtm <- TermDocumentMatrix(doc.corpus, control)
  return(doc.dtm)
}

spam.dtm <- get.tdm(all.spam)

spam.matrix <- as.matrix(spam.dtm)

spam.counts <- rowSums(spam.matrix)

spam.df <- data.frame(cbind(names(spam.counts), as.numeric(spam.counts)), 
                      stringsAsFactors = FALSE)

names(spam.df) <- c("term", "frequency")

spam.df$frequency <- as.numeric(spam.df$frequency)

spam.occurrence <- sapply(1:nrow(spam.matrix), function(i) {
  length(which(spam.matrix[i , ] > 0)) / ncol(spam.matrix)
})

spam.density <- spam.df$frequency / sum(spam.df$frequency)

spam.df <- transform(spam.df, density = spam.density, 
                     occurrence = spam.occurrence)


easyham.docs <- dir(easyham.path)
easyham.docs <- easyham.docs[which(easyham.docs != "cmds")]
all.easyham <- sapply(easyham.docs[1:length(spam.docs)], function(p) get.msg(paste(easyham.path, p, sep = "/")))
all.easyham <- iconv(all.easyham,"WINDOWS-1252","UTF-8")

easyham.dtm <- get.tdm(all.easyham)

easyham.matrix <- as.matrix(easyham.dtm)

easyham.counts <- rowSums(easyham.matrix)

easyham.df <- data.frame(cbind(names(easyham.counts), as.numeric(easyham.counts)), 
                      stringsAsFactors = FALSE)

names(easyham.df) <- c("term", "frequency")

easyham.df$frequency <- as.numeric(easyham.df$frequency)

easyham.occurrence <- sapply(1:nrow(easyham.matrix), function(i) {
  length(which(easyham.matrix[i , ] > 0)) / ncol(easyham.matrix)
})

easyham.density <- easyham.df$frequency / sum(easyham.df$frequency)

easyham.df <- transform(easyham.df, density = easyham.density, 
                     occurrence = easyham.occurrence)

classify.email <- function(path, training.df, prior = 0.5, c = 1e-6) {
  msg <- get.msg(path)
  msg <- iconv(msg,"WINDOWS-1252","UTF-8")
  msg.tdm <- get.tdm(msg)
  msg.matrix <- as.matrix(msg.tdm)
  msg.freq <- rowSums(msg.matrix)
  msg.match <- intersect(names(msg.freq), training.df$term)
  if(length(msg.match) < 1) {
    return(prior * c^(length(msg.freq)))
  }
  else {
    match.probs <- training.df$occurrence[match(msg.match, training.df$term)]
    return(prior * prod(match.probs) * c ^(length(msg.freq) - length(msg.match)))
  }
}

hardham.docs <- dir(hardham.path)
hardham.docs <- hardham.docs[which(hardham.docs != "cmds")]

hardham.spamtest <- sapply(hardham.docs, function(p) 
  classify.email(file.path(hardham.path, p), 
                 training.df = spam.df))

hardham.hamtest <- sapply(hardham.docs, function(p) 
  classify.email(file.path(hardham.path, p), 
                 training.df = easyham.df))

hardham.res <- ifelse(hardham.spamtest > hardham.hamtest, TRUE, FALSE)
summary(hardham.res)



spam.spamtest <- sapply(spam.docs, function(p) 
  classify.email(file.path(spam.path, p), 
                 training.df = spam.df))

spam.hamtest <- sapply(spam.docs, function(p) 
  classify.email(file.path(spam.path, p), 
                 training.df = easyham.df))

spam.res <- ifelse(spam.spamtest > spam.hamtest, TRUE, FALSE)
summary(spam.res)
