package ma.Mouhssine;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.Mouhssine.llm.llmClient;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class RagNaif {

    public static void main(String[] args) {

        try {
            Path path = Paths.get("src/main/resources/rag.pdf");
            System.out.println("Chargement du fichier : " + path);
            System.out.println("Fichier existe ? " + Files.exists(path));


            ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
            Document document = FileSystemDocumentLoader.loadDocument(path, parser);
            System.out.println("Document chargé avec succès.");


            System.out.println("######################################################");
            System.out.println("Texte du document : " + document.text().substring(0, Math.min(200, document.text().length())));
            System.out.println("######################################################");

            DocumentSplitter splitter = DocumentSplitters.recursive(500, 50);
            List<TextSegment> segments = splitter.split(document);
            System.out.println("Nombre de segments créés : " + segments.size());

            EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
            List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
            System.out.println("Embeddings générés : " + embeddings.size());

            EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
            embeddingStore.addAll(embeddings, segments);
            System.out.println("Embeddings enregistrés dans le store en mémoire.");

            var retriever = EmbeddingStoreContentRetriever.builder()
                    .embeddingStore(embeddingStore)
                    .embeddingModel(embeddingModel)
                    .maxResults(2)
                    .minScore(0.5)
                    .build();

            List<Content> found = retriever.retrieve(Query.from("c'est quoi le RAG ?"));
            System.out.println("Segments trouvés : " + found.size());



            String llmKey = System.getenv("GEMINI_API_KEY");
            if (llmKey == null) {
                System.err.println("GEMINIKEY non définie !");
                return;
            }

            // 2. Création du modèle LLM (Gemini)
            ChatModel model = GoogleAiGeminiChatModel.builder()
                    .apiKey(llmKey)
                    .modelName("gemini-2.5-flash") // modèle rapide et récent
                    .temperature(0.3)              // température faible = réponses plus précises
                    .maxOutputTokens(512)          // limite du nombre de tokens générés
                    .build();




            Assistant assistant = AiServices.builder(Assistant.class)
                    .chatModel(model)
                    .chatMemory(MessageWindowChatMemory.withMaxMessages(10)) // mémoire pour 10 messages
                    .contentRetriever(retriever)                            // le RAG retriever
                    .build();

            conversationAvec(assistant);


        } catch (Exception e) {
            e.printStackTrace();
        }

    }
    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question (ou tapez 'fin' pour quitter) : ");
                String question = scanner.nextLine();

                if (question.isBlank()) {
                    continue; // ignore les lignes vides
                }

                if ("fin".equalsIgnoreCase(question)) {
                    System.out.println("Conversation terminée.");
                    break;
                }

                System.out.println("==================================================");
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }
        }
}
