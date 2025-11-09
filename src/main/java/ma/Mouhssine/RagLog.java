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
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RagLog {

    public static void main(String[] args) throws Exception {


        configureLogger();


        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null) {
            System.err.println("Variable d'environnement GEMINI_API_KEY non définie !");
            return;
        }


        ChatModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.0-flash-exp")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();


        URL fileUrl = RagLog.class.getResource("/rag.pdf");
        if (fileUrl == null) {
            System.err.println("⚠️  Fichier 'support_rag.pdf' introuvable dans resources/");
            return;
        }
        Path filePath = Paths.get(fileUrl.toURI());


        var parser = new ApacheTikaDocumentParser();
        Document document = FileSystemDocumentLoader.loadDocument(filePath, parser);


        var splitter = DocumentSplitters.recursive(600, 100);
        List<TextSegment> segments = splitter.split(document);


        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();


        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);


        System.out.println("\n📘 Segments enregistrés :");
        for (int i = 0; i < Math.min(segments.size(), 5); i++) {
            String preview = segments.get(i).text();
            System.out.println("- " + preview.substring(0, Math.min(preview.length(), 100)) + "...");
        }


        ChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);


        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(chatModel)
                .chatMemory(memory)
                .build();


        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("\n🤖 Assistant RAG avec Logging activé — tapez 'fin' pour quitter.\n");

            while (true) {
                System.out.print("Vous : ");
                String question = scanner.nextLine();

                if (question.trim().equalsIgnoreCase("fin")) {
                    System.out.println("👋 Fin de la session.");
                    break;
                }

                if (question.isBlank()) continue;

                String reponse = assistant.chat(question);
                System.out.println("\nAssistant : " + reponse + "\n");
            }
        }
    }


    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        logger.addHandler(handler);
    }


    interface Assistant {
        String chat(String message);
    }
}