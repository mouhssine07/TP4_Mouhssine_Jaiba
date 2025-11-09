package ma.Mouhssine;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class Test5RagWeb {

    public static void main(String[] args) throws Exception {
        configureLogger();

        String tavilyKey1 = System.getenv("TAVILYKEY");
        if (tavilyKey1 == null) {
            System.out.println("null"); // valeur de secours
        }

        // --- Clés API ---
        String geminiKey = System.getenv("GEMINI_API_KEY");
        String tavilyKey = System.getenv("TAVILYKEY"); // clé Tavily pour la recherche Web
        if (geminiKey == null || tavilyKey == null) {
            System.err.println("Variables GEMINIKEY ou TAVILYKEY non définies !");
            return;
        }

        // --- Modèle Gemini ---
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // --- Ingestion du PDF ---
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        URL fileUrl = Test5RagWeb.class.getResource("/rag.pdf");
        Path path = Paths.get(fileUrl.toURI());
        Document doc = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(600,0).split(doc);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        ContentRetriever pdfRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.1)
                .build();

        // --- Web Search avec Tavily ---
        TavilyWebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(3)
                .build();

        // --- QueryRouter avec les 2 retrievers ---
        QueryRouter queryRouter = new DefaultQueryRouter(pdfRetriever, webRetriever);

        // --- RetrievalAugmentor ---
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // --- ChatMemory ---
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // --- Création de l'assistant ---
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // --- Interaction utilisateur ---
        Scanner scanner = new Scanner(System.in);
        System.out.println("Assistant RAG + WebSearch (PDF + Web). Tapez 'fin' pour quitter.\n");

        while (true) {
            System.out.print("Question : ");
            String question = scanner.nextLine();
            if ("fin".equalsIgnoreCase(question.trim())) break;
            if (question.trim().isEmpty()) continue;
            String reponse = assistant.chat(question.trim());
            System.out.println("Réponse : " + reponse + "\n");
        }
        scanner.close();
    }

    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }
}