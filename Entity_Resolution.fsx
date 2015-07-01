(*
Text Analysis and Entity Resolution

Entity resolution is a common, yet difficult problem in data cleaning and integration. 
Entity Resolution (ER) refers to the task of finding records in a dataset that refer to the same entity across different 
data sources (e.g., data files, books, websites, databases). ER is necessary when joining datasets based on entities that 
may or may not share a common identifier (e.g., database key, URI, National identification number), as may be the case due 
to differences in record shape, storage location, and/or curator style or preference. A dataset that has undergone ER may 
be referred to as being cross-linked.

Files:
The data files are from the metric-learning project and can be found on GitHub at:
https://github.com/gy8/learn_data-science/find/master

We are using the following files:
Google_small.csv, 200 records sampled from the Google data
Amazon_small.csv, 200 records sampled from the Amazon data
stopwords.txt, a list of common English words 
*)

#r @"..\packages\FSharp.Data.2.2.3\lib\net40\FSharp.Data.dll"
open FSharp.Data
open System.Text.RegularExpressions
open System.Collections.Generic

type GoogleSmall = CsvProvider< @"C:\Google_small.csv", HasHeaders = true, Schema="string,string,string,string,string">
let googleSmall = GoogleSmall.Load(@"C:\Google_small.csv")
type AmazonSmall = CsvProvider< @"C:\Amazon_small.csv", HasHeaders = true, Schema="string,string,string,string,string">
let amazonSmall = AmazonSmall.Load(@"C:\Amazon_small.csv")
let stopwords = System.IO.File.ReadAllLines(@"C:\stopwords.txt") |> Set.ofArray

[<Literal>]
let quickbrownfox = "A quick brown fox jumps over the lazy dog."

// A simple implementation of input string tokenization
let simpleTokenize (str:string) =
    let result = Regex.Matches(str.ToLower(), @"\b(\w+?)\b")
    [ for token in result -> (string)token ]

simpleTokenize quickbrownfox

// An implementation of input string tokenization that excludes stopwords
let tokenize (str: string) =
    [ for token in Regex.Matches(str.ToLower(), @"\b(\w+?)\b") do
        let s = (string)token
        if not (stopwords.Contains s) then yield s ]

tokenize quickbrownfox

let amazonRecToToken = amazonSmall.Rows |> Seq.map (fun r -> (r.Id, tokenize (sprintf "%s %s %s" r.Title r.Manufacturer r.Description)))
let googleRecToToken = googleSmall.Rows |> Seq.map (fun r -> (r.Id, tokenize (sprintf "%s %s %s" r.Name r.Manufacturer r.Description)))

// Count and return the number of tokens
let countTokens (vendorTokens: seq<string*string list>) =
    vendorTokens |> Seq.fold (fun acc (id,tokenList) -> acc + tokenList.Length) 0

// 22520
let totalTokens = countTokens amazonRecToToken + countTokens googleRecToToken

// Find and return the record with the largest number of tokens
let findBiggestRecord (vendorTokens: seq<string*string list>) =
    let mutable maxId = ""
    let mutable maxCount = 0
    vendorTokens |> Seq.iter (fun (id,tokenList) -> if tokenList.Length > maxCount then 
                                                        maxId <- id
                                                        maxCount <- tokenList.Length)
    (maxId, maxCount)

// "b000o24l3q"
findBiggestRecord amazonRecToToken

// Compute Term-Frequency (TF)
let tf (tokens: string list) =
    let tokenDict = new Dictionary<string, float>()
    let oneToken = 1. / (float)tokens.Length
    tokens
    |> List.iter (fun tok -> 
        if tokenDict.ContainsKey(tok) then tokenDict.[tok] <- tokenDict.[tok] + oneToken
        else tokenDict.Add(tok, oneToken))
    tokenDict |> Seq.toList

tf (tokenize "one_ one_ two!")

let corpus = Seq.append amazonRecToToken googleRecToToken

// Compute Inverse-Document-Frequency (IDF)
let idfs (corpus: seq<string*string list>) =
    let N = (float) (Seq.length corpus)
    let uniqueTokens = new Dictionary<string,float>()
    corpus |> Seq.iter (fun (id, tokenList) -> 
        tokenList
        |> Set.ofList
        |> Set.iter (fun tok -> 
            if uniqueTokens.ContainsKey(tok) then uniqueTokens.[tok] <- uniqueTokens.[tok] + 1.
            else uniqueTokens.Add(tok, 1.)))
    uniqueTokens.Keys 
    |> Seq.toList
    |> List.iter (fun tok -> uniqueTokens.[tok] <- N / uniqueTokens.[tok])
    uniqueTokens

let idfsSmallWeights = idfs corpus
let tokenSmallestIdf = idfsSmallWeights |> Seq.minBy (fun kvp -> kvp.Value) 
// 11 tokens with the smallest IDF in the combined small dataset
let smallIDFTokens = idfsSmallWeights |> Seq.sortBy (fun kvp -> kvp.Value) |> Seq.take 11

// Plot a histogram of IDF values
#load "..\packages\FSharp.Charting.0.90.10\FSharp.Charting.fsx"
open FSharp.Charting

let bins = Array.zeroCreate<int> 50
let minIdf = idfsSmallWeights |> Seq.minBy (fun kvp -> kvp.Value)
let maxIdf = idfsSmallWeights |> Seq.maxBy (fun kvp -> kvp.Value)
let binSize = ceil (maxIdf.Value - floor minIdf.Value) / 50.
idfsSmallWeights |> Seq.iter (fun kvp -> 
    let binNo = int (truncate ((kvp.Value - minIdf.Value) / binSize))
    bins.[binNo] <- bins.[binNo] + 1)
Chart.Column bins

// Compute Term-Frequency*Inverse-Document-Frequency (TF-IDF)
let tfidf (tokens: string list) (idfs: Dictionary<string,float>) =
    let tfs = tf tokens
    let tfIdfDict = new Dictionary<string,float>()
    tfs |> Seq.iter (fun kvp -> tfIdfDict.Add(kvp.Key,kvp.Value * idfs.[kvp.Key]))
    tfIdfDict

// Amazon record "b000hkgj8k" has tokens and weights:
let recb000hkgj8k = amazonRecToToken |> Seq.find (fun (id, tokenList) -> id = "b000hkgj8k") |> snd
let rec_b000hkgj8k_weights = tfidf recb000hkgj8k idfsSmallWeights

// Compute dot product
let dotprod (a: Dictionary<string,float>) (b: Dictionary<string,float>) =
    Set.intersect (Set.ofSeq a.Keys) (Set.ofSeq b.Keys)
    |> Set.fold (fun acc tok -> acc + a.[tok] * b.[tok]) 0.
 
// Compute square root of the dot product
let norm (a: Dictionary<string,float>) =
    a
    |> Seq.fold (fun acc kvp -> acc + kvp.Value * kvp.Value) 0.
    |> sqrt

// Compute cosine similarity
let cossim (a: Dictionary<string,float>) (b: Dictionary<string,float>) =
    (dotprod a b) / ((norm a) * (norm b))

let testVec1 = new Dictionary<string,float>()
testVec1.Add("foo", 2.)
testVec1.Add("bar", 3.)
testVec1.Add("baz", 5.)
let testVec2 = new Dictionary<string,float>()
testVec2.Add("foo", 1.)
testVec2.Add("bar", 0.)
testVec2.Add("baz", 20.)
dotprod testVec1 testVec2
norm testVec1

// Compute cosine similarity between two strings
let cosineSimilarity string1 string2 (idfs: Dictionary<string,float>) =
    cossim (tfidf (tokenize string1) idfs) (tfidf (tokenize string2) idfs)

// 0.05772433822
cosineSimilarity "Adobe Photoshop" "Adobe Illustrator" idfsSmallWeights

// Compute similarity on a combination record
let computeSimilarity ((googleUrl, googleValue), (amazonID, amazonValue)) =
    (googleUrl, amazonID, cosineSimilarity googleValue amazonValue idfsSmallWeights)

// Return similarity value
let similar (amazonID:string) (googleUrl: string) (idfs: Dictionary<string,float>) =
    let (amazonId, amazonValue) = amazonRecToToken |> Seq.find (fun (id, _) -> id = amazonID)
    let (googleUrl, googleValue) = googleRecToToken |> Seq.find (fun (id, _) -> id = googleUrl)
    cossim (tfidf amazonValue idfs) (tfidf googleValue idfs) 

// 0.0003031719405
similar "b000o24l3q" "http://www.google.com/base/feeds/snippets/17242822440574356561" idfsSmallWeights
