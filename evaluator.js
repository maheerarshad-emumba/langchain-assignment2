import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createInterface } from "readline";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { loadEvaluator } from "langchain/evaluation";
import path from 'path';
import * as dotenv from "dotenv";

dotenv.config();

//initialize OpenAI model
const initializeModel = () => {
    return new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
        temperature: 0.7,
    });
};

//load and process PDF document
const loadAndProcessPDF = async (pdfPath) => {
    const pdfLoader = new PDFLoader(pdfPath);
    const docs = await pdfLoader.load();
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20,
    });
    return await splitter.splitDocuments(docs);
};

//load and process webpage content
const loadAndProcessWebPage = async (url) => {
    const webpageLoader = new CheerioWebBaseLoader(url);
    const docs = await webpageLoader.load();
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20,
    });
    return await splitter.splitDocuments(docs);
};

//create a vector store from the split documents
const createVectorStore = async (splitDocs) => {
    const embeddings = new OpenAIEmbeddings();
    return await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
};

//create retrieval chain
const createRetrievalChainFromDocs = async (model, vectorstore) => {
    const retriever = vectorstore.asRetriever();
    const prompt = ChatPromptTemplate.fromMessages([
        { role: "system", content: "Context: {context}. Answer the user's question based on the context provided. If the question is not related to the context, respond with 'I do not have information about it.'" },
        new MessagesPlaceholder("chat_history"),
        { role: "human", content: "{input}" },
    ]);

    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt,
    });

    return await createRetrievalChain({
        combineDocsChain: chain,
        retriever,
    });
};

//set up readline interface for user input
const setupReadlineInterface = () => {
    return createInterface({
        input: process.stdin,
        output: process.stdout,
    });
};

//evaluator for cosine distance
const evaluator = await loadEvaluator("embedding_distance", {
  distanceMetric: "cosine",
});


const sampleQA = [
  {
    input: "How much loan can I take?",
    reference: "You can take a loan of up to 3 times your monthly salary."
  },
  {
    input: "What services does Emumba offer?",
    reference: "Emumba offers software development, AI solutions, and cloud infrastructure."
  }
];

//evaluate using embeddings
const evaluatePrediction = async (model, prediction, input) => {
    const referenceQA = sampleQA.find(qa => qa.input === input);
    if (!referenceQA) return;

    const res = await evaluator.evaluateStrings({
        prediction: prediction,
        reference: referenceQA.reference
    });

    console.log(`Input: ${input}`);
    console.log(`Prediction: ${prediction}`);
    console.log(`Expected: ${referenceQA.reference}`);
    console.log(`Evaluation Score: ${res.score}`);

    await compareResponsesWithPrompt(model, input, prediction, referenceQA.reference);
};

//comparison template
const comparisonPrompt = ChatPromptTemplate.fromTemplate(`
  The user asked: {input}.
  The chatbot's response was: {prediction}.
  The correct response is: {reference}.
  
  Please provide a score between 0 and 1 on how close the chatbot's response is to the correct one, followed by a brief explanation.
`);

//compare using prompt-based validation
const compareResponsesWithPrompt = async (model, input, prediction, reference) => {
    const promptMessage = await comparisonPrompt.formatMessages({
      input: input,
      prediction: prediction,
      reference: reference,
    });

    const response = await model.call(promptMessage);
  
    console.log(`Prompt-based Evaluation: ${response.content}`);
  };



const askQuestion = (model, rl, retrievalChain, chatHistory) => {
    rl.question("User: ", async(input) => {
        if (input.toLowerCase() === "exit"){
            rl.close();
            return;
        }

        const response = await retrievalChain.invoke({
            input: input,
            chat_history: chatHistory,
        });

        console.log("Agent: ", response.answer);
        chatHistory.push(new HumanMessage(input));
        chatHistory.push(new AIMessage(response.answer));

        const prediction = response.answer;
        await evaluatePrediction(model, prediction, input);

        askQuestion(model, rl, retrievalChain, chatHistory);
    });
};

const runChatbot = async () => {
    const model = initializeModel();

    const pdfPath = path.resolve("C:/Users/Emumba/Documents/qa pathways/assignment1 langchain/Loan Policy of Emumba Inc.pdf");
    const pdfDocs = await loadAndProcessPDF(pdfPath);

    const webDocs = await loadAndProcessWebPage("https://emumba.com/");
    const allDocs = [...pdfDocs, ...webDocs];

    const vectorstore = await createVectorStore(allDocs);
    const retrievalChain = await createRetrievalChainFromDocs(model, vectorstore);

    const rl = setupReadlineInterface();
    const chatHistory = [];

    askQuestion(model, rl, retrievalChain, chatHistory);
};

runChatbot();
