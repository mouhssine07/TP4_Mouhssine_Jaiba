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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {

    public static void main(String[] args) throws Exception {
        configureLogger();

        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null) {
            System.err.println("Variable d'environnement GEMINIKEY non définie !");
            return;
        }

        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // --- Création des ContentRetrievers pour les 2 PDFs ---
        ContentRetriever retrieverIA = buildRetriever("/rag.pdf");
        ContentRetriever retrieverGestion = buildRetriever("/CTRL1_5_MICRO_SERVICES_CONFIG_SERVER.pdf");

        // --- Map de description pour le QueryRouter ---
        Map<ContentRetriever, String> descriptions = new LinkedHashMap<>();
        descriptions.put(retrieverIA, "Support sur l’IA, RAG, LangChain4j, modèles de langage.");
        descriptions.put(retrieverGestion, "Cours sur microservices et configuration serveur, non lié à l’IA.");

        QueryRouter router = new LanguageModelQueryRouter(chatModel, descriptions);

        // --- Création du RetrievalAugmentor avec le QueryRouter ---
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // --- Assistant avec retrievalAugmentor ---
        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentor)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("\nAssistant Routage TP3. Tapez 'fin' pour quitter.\n");

        while (true) {
            System.out.print("Question : ");
            String question = scanner.nextLine();
            if ("fin".equalsIgnoreCase(question.trim())) break;
            if (question.trim().isEmpty()) continue;

            // --- Appel de l'assistant avec routage automatique ---
            String reponse = assistant.chat(question);
            System.out.println("\nRéponse Gemini : " + reponse + "\n");
        }

        scanner.close();
    }

    // --- Méthode pour construire un ContentRetriever à partir d'un PDF ---
    private static ContentRetriever buildRetriever(String resourceName) throws Exception {
        URL fileUrl = TestRoutage.class.getResource(resourceName);
        if (fileUrl == null) throw new RuntimeException("Fichier introuvable : " + resourceName);
        Path path = Paths.get(fileUrl.toURI());

        Document doc = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(500, 0).split(doc);

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        System.out.println("Document '" + resourceName + "' ingéré. Nb segments : " + segments.size());
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.1)
                .build();
    }

    // --- Configuration du logger ---
    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }
}