package main

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/ledongthuc/pdf"
)

// Strutture dati
type Document struct {
	ID      string    `json:"id"`
	Content string    `json:"content"`
	Page    int       `json:"page"`
	Vector  []float64 `json:"vector"`
}

type VectorStore struct {
	Documents []Document `json:"documents"`
	ModelName string     `json:"model_name"`
}

type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type OllamaRequest3T struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	Temperature float64 `json:"temperature,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
}

type OllamaResponse struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

type EmbeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type EmbeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

type RAGChatbot struct {
	vectorStore   *VectorStore
	ollamaBaseURL string
	embedModel    string
	chatModel     string
	dbPath        string
}

// Inizializza il chatbot
func NewRAGChatbot() *RAGChatbot {
	return &RAGChatbot{
		vectorStore:   &VectorStore{Documents: []Document{}},
		ollamaBaseURL: "http://localhost:11434",
		embedModel:    "mistral", //"nomic-embed-text", // Modello di embedding
		chatModel:     "",        // Modello di chat
		dbPath:        "vectorstore.json",
	}
}

// Estrae testo dal PDF
func (r *RAGChatbot) ExtractTextFromPDF(filename string) ([]string, error) {
	file, reader, err := pdf.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("errore apertura PDF: %v", err)
	}
	defer file.Close()

	var pages []string
	totalPages := reader.NumPage()

	fmt.Printf("Elaborazione PDF: %d pagine trovate\n", totalPages)

	for pageNum := 1; pageNum <= totalPages; pageNum++ {
		page := reader.Page(pageNum)
		if page.V.IsNull() {
			continue
		}

		content, err := page.GetPlainText(nil)
		if err != nil {
			log.Printf("Errore estrazione pagina %d: %v", pageNum, err)
			continue
		}

		// Pulisci il testo
		cleanContent := r.cleanText(content)
		if len(cleanContent) > 50 { // Solo se ha contenuto significativo
			pages = append(pages, cleanContent)
		}
	}

	return pages, nil
}

// Pulisce il testo estratto
func (r *RAGChatbot) cleanText(text string) string {
	// Rimuovi caratteri di controllo e normalizza spazi
	reg := regexp.MustCompile(`\s+`)
	text = reg.ReplaceAllString(text, " ")

	// Rimuovi caratteri non stampabili
	reg = regexp.MustCompile(`[^\p{L}\p{N}\p{P}\p{Z}]+`)
	text = reg.ReplaceAllString(text, " ")

	return strings.TrimSpace(text)
}

// Suddivide il testo in chunks
func (r *RAGChatbot) ChunkText(text string, chunkSize int, overlap int) []string {
	words := strings.Fields(text)
	if len(words) <= chunkSize {
		return []string{text}
	}

	var chunks []string
	for i := 0; i < len(words); i += chunkSize - overlap {
		end := i + chunkSize
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[i:end], " ")
		chunks = append(chunks, chunk)

		if end == len(words) {
			break
		}
	}

	return chunks
}

// Genera embedding tramite Ollama
func (r *RAGChatbot) GetEmbedding(text string) ([]float64, error) {
	reqBody := EmbeddingRequest{
		Model: r.embedModel,
		Input: text,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(r.ollamaBaseURL+"/api/embed", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("errore chiamata Ollama embed: %v", err)
	}
	defer resp.Body.Close()

	var embedResp EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, err
	}

	if len(embedResp.Embeddings) == 0 {
		return nil, fmt.Errorf("nessun embedding ricevuto")
	}

	return embedResp.Embeddings[0], nil
}

// Calcola similarità coseno
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (normA * normB)
}

// Elabora PDF e crea vector store
func (r *RAGChatbot) ProcessPDF(filename string) error {
	fmt.Println("📄 Estrazione testo dal PDF...")
	pages, err := r.ExtractTextFromPDF(filename)
	if err != nil {
		return err
	}

	fmt.Printf("✅ Estratte %d pagine\n", len(pages))

	r.vectorStore.Documents = []Document{}

	fmt.Println("🔤 Creazione chunks e embedding...")
	totalChunks := 0

	for pageNum, pageText := range pages {
		// Crea chunks per ogni pagina
		chunks := r.ChunkText(pageText, 100, 20)
		// 300 token , 50 overlap
		// 100 token, 20 overlap

		for chunkIdx, chunk := range chunks {
			if len(strings.TrimSpace(chunk)) < 20 {
				continue
			}

			// Genera ID unico
			hasher := md5.New()
			hasher.Write([]byte(chunk))
			docID := fmt.Sprintf("page_%d_chunk_%d_%x", pageNum+1, chunkIdx, hasher.Sum(nil)[:4])

			fmt.Printf("🔄 Processando chunk %d/%d (pagina %d)\r", totalChunks+1, len(chunks), pageNum+1)

			// Genera embedding
			vector, err := r.GetEmbedding(chunk)
			if err != nil {
				log.Printf("Errore embedding per chunk %s: %v", docID, err)
				continue
			}

			doc := Document{
				ID:      docID,
				Content: chunk,
				Page:    pageNum + 1,
				Vector:  vector,
			}

			r.vectorStore.Documents = append(r.vectorStore.Documents, doc)
			totalChunks++

			// Pausa per non sovraccaricare Ollama
			time.Sleep(100 * time.Millisecond)
		}
	}

	r.vectorStore.ModelName = r.embedModel
	fmt.Printf("\n✅ Creati %d chunks con embedding\n", totalChunks)

	return r.SaveVectorStore()
}

// Salva vector store su file
func (r *RAGChatbot) SaveVectorStore() error {
	data, err := json.MarshalIndent(r.vectorStore, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(r.dbPath, data, 0644)
}

// Carica vector store da file
func (r *RAGChatbot) LoadVectorStore() error {
	if _, err := os.Stat(r.dbPath); os.IsNotExist(err) {
		return fmt.Errorf("database non esistente")
	}

	data, err := os.ReadFile(r.dbPath)
	if err != nil {
		return err
	}

	return json.Unmarshal(data, r.vectorStore)
}

// Ricerca documenti simili
func (r *RAGChatbot) SearchSimilar(query string, topK int) ([]Document, error) {
	queryVector, err := r.GetEmbedding(query)
	if err != nil {
		return nil, err
	}

	type ScoredDocument struct {
		Document Document
		Score    float64
	}

	var scoredDocs []ScoredDocument

	for _, doc := range r.vectorStore.Documents {
		similarity := cosineSimilarity(queryVector, doc.Vector)
		scoredDocs = append(scoredDocs, ScoredDocument{
			Document: doc,
			Score:    similarity,
		})
	}

	// Ordina per similarità decrescente
	sort.Slice(scoredDocs, func(i, j int) bool {
		return scoredDocs[i].Score > scoredDocs[j].Score
	})

	// Prendi i top K
	if topK > len(scoredDocs) {
		topK = len(scoredDocs)
	}

	var results []Document
	for i := 0; i < topK; i++ {
		results = append(results, scoredDocs[i].Document)
	}

	return results, nil
}

// Genera risposta tramite Ollama
func (r *RAGChatbot) GenerateResponse(question string, context []Document) (string, error) {
	// Costruisci il contesto
	var contextText strings.Builder
	contextText.WriteString("Contesto dal documento:\n\n")

	for i, doc := range context {
		contextText.WriteString(fmt.Sprintf("Sezione %d (Pagina %d):\n%s\n\n", i+1, doc.Page, doc.Content))
	}

	// Prompt ottimizzato per l'italiano
	prompt := fmt.Sprintf(`Sei un assistente che risponde a domande basandoti esclusivamente sul documento fornito.

%s

Domanda: %s

Istruzioni:
- Rispondi SOLO in italiano
- Usa ESCLUSIVAMENTE le informazioni del contesto fornito
- Se la risposta non è presente nel documento, dillo chiaramente
- Sii preciso e dettagliato
- Cita la pagina quando possibile

Risposta:`, contextText.String(), question)

	// default
	/*reqBody := OllamaRequest{
		Model:  r.chatModel,
		Prompt: prompt,
		Stream: false,
	}*/

	reqBody := OllamaRequest3T{
		Model:  r.chatModel,
		Prompt: prompt,
		Stream: false,
		//		Temperature: 0.2, // Bassa temperatura per risposte più precise e consistenti
		//		TopK:        40,  // Limita le opzioni di token
		//		TopP:        0.9, // Nucleus sampling
		Temperature: 0.4,  // Aumentata per più creatività nella riformulazione
		TopK:        60,   // Più opzioni per linguaggio naturale
		TopP:        0.95, // Maggiore varietà lessicale
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	resp, err := http.Post(r.ollamaBaseURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("errore chiamata Ollama: %v", err)
	}
	defer resp.Body.Close()

	var response OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return "", err
	}

	return response.Response, nil
}

// Chat con il documento
func (r *RAGChatbot) Chat(question string) (string, []Document, error) {
	if len(r.vectorStore.Documents) == 0 {
		return "Per favore, carica prima un documento PDF.", nil, nil
	}

	// Cerca documenti simili
	similarDocs, err := r.SearchSimilar(question, 4)
	if err != nil {
		return "", nil, err
	}

	// Genera risposta
	answer, err := r.GenerateResponse(question, similarDocs)
	if err != nil {
		return "", nil, err
	}

	return answer, similarDocs, nil
}

// Versione alternativa con riscrittura esplicita
func (r *RAGChatbot) GenerateNaturalResponse(question string, context []Document) (string, error) {
	// Prima estrai le informazioni chiave
	var contextText strings.Builder
	contextText.WriteString("Informazioni dal documento:\n")

	for i, doc := range context {
		contextText.WriteString(fmt.Sprintf("- Fonte %d: %s\n", i+1, doc.Content))
	}

	// Prompt in due fasi per evitare copia letterale
	analysisPrompt := fmt.Sprintf(`Analizza le seguenti informazioni e identifica i punti chiave per rispondere alla domanda.

INFORMAZIONI:
%s

DOMANDA: %s

Elenca i punti chiave e i concetti principali senza copiare frasi intere:`, contextText.String(), question)

	// Prima fase: analisi
	analysisBody := OllamaRequest3T{
		Model:       r.chatModel,
		Prompt:      analysisPrompt,
		Stream:      false,
		Temperature: 0.3,
		TopK:        40,
		TopP:        0.9,
	}

	jsonData, err := json.Marshal(analysisBody)
	if err != nil {
		return "", err
	}

	log.Printf("Invio questo prompt: %d", analysisPrompt)
	resp, err := http.Post(r.ollamaBaseURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var analysisResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&analysisResp); err != nil {
		return "", err
	}

	// Seconda fase: riscrittura naturale
	rewritePrompt := fmt.Sprintf(`Basandoti sui seguenti punti chiave, scrivi una risposta naturale e fluida in italiano.

PUNTI CHIAVE IDENTIFICATI:
%s

DOMANDA ORIGINALE: %s

ISTRUZIONI:
- Scrivi con parole tue, non copiare
- Usa un linguaggio naturale e conversazionale
- Organizza le informazioni in modo logico
- Aggiungi transizioni fluide tra i concetti
- Se utile, usa esempi o analogie
- Mantieni un tono professionale ma accessibile

Risposta naturale:`, analysisResp.Response, question)

	rewriteBody := OllamaRequest3T{
		Model:       r.chatModel,
		Prompt:      rewritePrompt,
		Stream:      false,
		Temperature: 0.6, // Più alta per maggiore naturalezza
		TopK:        80,
		TopP:        0.95,
	}

	jsonData, err = json.Marshal(rewriteBody)
	if err != nil {
		return "", err
	}

	log.Printf("Invio prompt riscritto: %d", rewritePrompt)
	resp, err = http.Post(r.ollamaBaseURL+"/api/generate", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var finalResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&finalResp); err != nil {
		return "", err
	}

	return finalResp.Response, nil
}

// Verifica se Ollama è disponibile
func (r *RAGChatbot) CheckOllamaAvailable() error {
	resp, err := http.Get(r.ollamaBaseURL + "/api/tags")
	if err != nil {
		return fmt.Errorf("Ollama non disponibile su %s: %v", r.ollamaBaseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("Ollama risponde con status: %d", resp.StatusCode)
	}

	return nil
}

// Elabora file TXT e crea vector store
func (r *RAGChatbot) ProcessTXT(filename string) error {
	fmt.Println("� Lettura file TXT...")

	// Leggi tutto il contenuto del file
	content, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("errore lettura file TXT: %v", err)
	}

	text := string(content)

	// Pulisci il testo
	cleanedText := r.cleanText(text)

	if len(strings.TrimSpace(cleanedText)) < 50 {
		return fmt.Errorf("file TXT troppo corto o vuoto")
	}

	fmt.Printf("✅ File letto: %d caratteri\n", len(cleanedText))

	// Reset del vector store
	r.vectorStore.Documents = []Document{}

	fmt.Println("� Creazione chunks e embedding...")

	// Crea chunks dal testo completo
	chunks := r.ChunkText(cleanedText, 300, 50) // 300 parole per chunk, overlap 50

	fmt.Printf("� Creati %d chunks\n", len(chunks))

	for chunkIdx, chunk := range chunks {
		if len(strings.TrimSpace(chunk)) < 20 {
			continue
		}

		// Genera ID unico per il chunk
		hasher := md5.New()
		hasher.Write([]byte(chunk))
		docID := fmt.Sprintf("txt_chunk_%d_%x", chunkIdx, hasher.Sum(nil)[:4])
		// TODO nel caso di problema di encoding per lettere accentate
		//docID := fmt.Sprintf("txt_%s_chunk_%d_%x", "ISO-8859-1", chunkIdx, hasher.Sum(nil)[:4])

		fmt.Printf("� Processando chunk %d/%d\r", chunkIdx+1, len(chunks))

		// Genera embedding
		vector, err := r.GetEmbedding(chunk)
		if err != nil {
			log.Printf("Errore embedding per chunk %s: %v", docID, err)
			continue
		}

		doc := Document{
			ID:      docID,
			Content: chunk,
			Page:    1, // Per file TXT usiamo sempre pagina 1
			Vector:  vector,
		}

		r.vectorStore.Documents = append(r.vectorStore.Documents, doc)

		// Pausa per non sovraccaricare Ollama
		time.Sleep(100 * time.Millisecond)
	}

	r.vectorStore.ModelName = r.embedModel
	fmt.Printf("\n✅ Creati %d chunks con embedding\n", len(r.vectorStore.Documents))

	return r.SaveVectorStore()
}

func main() {
	chatbot := NewRAGChatbot()

	fmt.Println("🤖 Chatbot RAG Offline per manuali Italiani")
	fmt.Println("=====================================")

	// Verifica Ollama
	fmt.Println("🔍 Verifica disponibilità Ollama...")
	if err := chatbot.CheckOllamaAvailable(); err != nil {
		log.Fatal("❌ ", err)
	}
	fmt.Println("✅ Ollama disponibile")

	// Prova a caricare database esistente
	fmt.Println("📂 Caricamento database esistente...")
	if err := chatbot.LoadVectorStore(); err != nil {
		fmt.Println("⚠️  Nessun database esistente trovato")
	} else {
		fmt.Printf("✅ Database caricato: %d documenti\n", len(chatbot.vectorStore.Documents))
	}

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("\n📋 Opzioni disponibili:")
		fmt.Println("1. Elabora nuovo manuale")
		fmt.Println("2. Fai una domanda")
		fmt.Println("3. Mostra statistiche database")
		fmt.Println("4. Esci")
		fmt.Print("\nScegli un'opzione (1-4): ")

		choice, _ := reader.ReadString('\n')
		choice = strings.TrimSpace(choice)

		switch choice {
		case "1":
			fmt.Print("\n📄 Inserisci il percorso del manuale: ")
			pdfPath, _ := reader.ReadString('\n')
			pdfPath = strings.TrimSpace(pdfPath)

			if _, err := os.Stat(pdfPath); os.IsNotExist(err) {
				fmt.Println("❌ File non trovato")
				continue
			}

			fmt.Println("\n🚀 Inizio elaborazione manuale...")
			start := time.Now()

			if err := chatbot.ProcessTXT(pdfPath); err != nil {
				//if err := chatbot.ProcessPDF(pdfPath); err != nil {
				fmt.Printf("❌ Errore: %v\n", err)
			} else {
				duration := time.Since(start)
				fmt.Printf("✅ Manuale elaborato con successo in %v\n", duration)
				fmt.Printf("📊 Documenti nel database: %d\n", len(chatbot.vectorStore.Documents))
			}

		case "2":
			if len(chatbot.vectorStore.Documents) == 0 {
				fmt.Println("⚠️  Carica prima un manuale!")
				continue
			}

			fmt.Print("\n❓ Inserisci la tua domanda: ")
			question, _ := reader.ReadString('\n')
			question = strings.TrimSpace(question)

			if question == "" {
				continue
			}

			fmt.Println("\n🤔 Sto pensando...")
			start := time.Now()

			// versione più fluida
			sources, err := chatbot.SearchSimilar(question, 2)
			if err != nil {
				fmt.Printf("❌ Errore ricerca: %v\n", err)
				continue
			}
			answer, err := chatbot.GenerateNaturalResponse(question, sources)

			// versione più aderente al manuale
			/*answer, sources, err := chatbot.Chat(question)
			if err != nil {
				fmt.Printf("❌ Errore: %v\n", err)
				continue
			}*/

			duration := time.Since(start)
			fmt.Printf("\n💬 Risposta (generata in %v):\n", duration)
			fmt.Printf("─────────────────────────────────\n")
			fmt.Println(answer)

			if len(sources) > 0 {
				fmt.Println("\n📚 Fonti utilizzate:")
				for i, source := range sources {
					fmt.Printf("\n🔹 Fonte %d (Pagina %d):\n", i+1, source.Page)
					preview := source.Content
					/*if len(preview) > 200 {
						preview = preview[:200] + "..."
					}*/
					fmt.Println(preview)
				}
			}

		case "3":
			fmt.Printf("\n📊 Statistiche Database:\n")
			fmt.Printf("─────────────────────\n")
			fmt.Printf("📄 Documenti totali: %d\n", len(chatbot.vectorStore.Documents))
			fmt.Printf("🔤 Modello embedding: %s\n", chatbot.vectorStore.ModelName)

			if len(chatbot.vectorStore.Documents) > 0 {
				// Calcola statistiche pagine
				pageCount := make(map[int]int)
				totalChars := 0
				for _, doc := range chatbot.vectorStore.Documents {
					pageCount[doc.Page]++
					totalChars += len(doc.Content)
				}

				fmt.Printf("📖 Pagine elaborate: %d\n", len(pageCount))
				fmt.Printf("📊 Caratteri totali: %d\n", totalChars)
				fmt.Printf("📏 Media caratteri per chunk: %.0f\n", float64(totalChars)/float64(len(chatbot.vectorStore.Documents)))
			}

		case "4":
			fmt.Println("\n👋 Arrivederci!")
			return

		default:
			fmt.Println("❌ Opzione non valida")
		}
	}
}
