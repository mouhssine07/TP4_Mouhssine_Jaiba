package ma.Mouhssine;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.UserMessage;
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
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

// … tous vos imports

public class Test4PasDeRag {

    public static void main(String[] args) throws Exception {
        configureLogger();

        String llmKey = System.getenv("GEMINI_API_KEY");
        if (llmKey == null) {
            System.err.println("Variable d'environnement GEMINIKEY non définie !");
            return;
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // Ingestion PDF etc …
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        URL fileUrl = Test4PasDeRag.class.getResource("/rag.pdf");
        Path path = Paths.get(fileUrl.toURI());
        Document doc = FileSystemDocumentLoader.loadDocument(path, new ApacheTikaDocumentParser());
        List<TextSegment> segments = DocumentSplitters.recursive(600, 0).split(doc);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(3)
                .minScore(0.1)
                .build();

        System.out.println("✅ Ingestion du PDF terminée ! Nombre de segments : " + segments.size());

        class QueryRouterPasRag implements QueryRouter {
            @Override
            public List<ContentRetriever> route(Query query) {
                String promptText = "Est‑ce que la requête '" + query.text() +
                        "' concerne l'IA, le RAG, les LLM ou le fine‑tuning ? Réponds par 'oui','non' ou 'peut‑être'.";
                UserMessage userMessage = new UserMessage(promptText);

                // Au lieu de .messages().get(0).content(), utilisez :
                var chatResponse = model.chat(userMessage);   // peut retourner un objet avec méthode .aiMessage() ou .text()
                String reponse = String.valueOf(chatResponse.aiMessage());   // ou .text(), selon version

                System.out.println("🔍 Décision routage : " + reponse.trim());

                if (reponse.trim().toLowerCase().startsWith("non")) {
                    return Collections.emptyList();
                } else {
                    return Collections.singletonList(contentRetriever);
                }
            }
        }

        QueryRouter queryRouter = new QueryRouterPasRag();

        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        ChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        // Création assistant sans systemMessage si non supporté
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("Assistant RAG filtré (Pas de RAG hors IA). Tapez 'fin' pour quitter.\n");

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
