package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/example/stablediffusion"
)

// GenerateRequest 直接复用文生图接口的 Go 结构体参数
type GenerateRequest = stablediffusion.Txt2ImgParams

type GenerateResponse struct {
	Images []ImageInfo `json:"images"`
	// Seed 本次生成实际使用的基准种子（请求传负数时为服务端随机生成）
	Seed  int64  `json:"seed,omitempty"`
	Error string `json:"error,omitempty"`
}

type ImageInfo struct {
	Width  uint32 `json:"width"`
	Height uint32 `json:"height"`
	// Seed 生成该图像实际使用的种子
	Seed   int64  `json:"seed"`
	Format string `json:"format"`
	Data   string `json:"data"` // Base64 编码的 PNG
}

// 全局变量
var (
	sdCtx *stablediffusion.Context
	// 并发控制：最多同时处理 1 个生成请求 (SD 通常是单流的)
	sem = make(chan struct{}, 1)
)

// 初始化函数
func init() {
	// 从环境变量获取模型路径
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "models/model.gguf"
	}

	// 创建上下文
	var err error
	options := stablediffusion.DefaultContextOptions(modelPath)
	sdCtx, err = stablediffusion.NewContext(options)
	if err != nil {
		log.Printf("Warning: Failed to create context: %v", err)
		log.Println("Server will start but image generation will fail without a valid model")
	}
}

// 健康检查端点
func healthCheckHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]string{
		"status":  "ok",
		"version": stablediffusion.GetVersion(),
		"commit":  stablediffusion.GetCommit(),
	}
	json.NewEncoder(w).Encode(response)
}

// 系统信息端点
func systemInfoHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]string{
		"system_info": stablediffusion.GetSystemInfo(),
		"version":     stablediffusion.GetVersion(),
		"commit":      stablediffusion.GetCommit(),
	}
	json.NewEncoder(w).Encode(response)
}

// 图像生成端点
func generateHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 解析请求
	var req GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}

	// 检查上下文是否初始化
	if sdCtx == nil {
		response := GenerateResponse{
			Error: "Context not initialized: missing model file",
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
		return
	}

	// 并发控制
	select {
	case sem <- struct{}{}:
		defer func() { <-sem }()
	default:
		http.Error(w, "Server is busy", http.StatusServiceUnavailable)
		return
	}

	// 并发控制
	select {
	case sem <- struct{}{}:
		defer func() { <-sem }()
	default:
		http.Error(w, "Server is busy", http.StatusServiceUnavailable)
		return
	}

	// 调用文生图接口：默认值填充、参数校验、seed 处理、PNG 编码均在 Go 层完成
	output, err := sdCtx.Txt2Img(req)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(GenerateResponse{
			Error: fmt.Sprintf("Failed to generate image: %v", err),
		})
		return
	}

	// 转换为响应格式
	imageInfos := make([]ImageInfo, len(output.Images))
	for i, img := range output.Images {
		imageInfos[i] = ImageInfo{
			Width:  img.Width,
			Height: img.Height,
			Seed:   img.Seed,
			Format: img.Format,
			Data:   base64.StdEncoding.EncodeToString(img.Data),
		}
	}

	// 返回响应
	response := GenerateResponse{
		Images: imageInfos,
		Seed:   output.Seed,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func main() {
	// 获取端口
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// 获取主机
	host := os.Getenv("HOST")
	if host == "" {
		host = "0.0.0.0"
	}

	mux := http.NewServeMux()
	// 注册路由
	mux.HandleFunc("/health", healthCheckHandler)
	mux.HandleFunc("/system-info", systemInfoHandler)

	// 给生成接口加上超时机制
	generateTimeout := 5 * time.Minute
	if t := os.Getenv("GENERATE_TIMEOUT"); t != "" {
		if d, err := time.ParseDuration(t); err == nil {
			generateTimeout = d
		}
	}

	generateTimeoutHandler := http.TimeoutHandler(http.HandlerFunc(generateHandler), generateTimeout, `{"error":"Generation timeout"}`)
	mux.Handle("/generate", generateTimeoutHandler)

	// 启动服务器
	serverAddr := fmt.Sprintf("%s:%s", host, port)
	srv := &http.Server{
		Addr:    serverAddr,
		Handler: mux,
	}

	log.Printf("Starting server on %s", serverAddr)
	log.Printf("Health check: http://localhost:%s/health", port)
	log.Printf("System info: http://localhost:%s/system-info", port)
	log.Printf("Generate endpoint: POST http://localhost:%s/generate", port)

	// 在后台启动服务器
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// 优雅停机处理
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("Shutting down server...")

	// 停止接收新的请求并等待现有请求完成（最多等待1分钟）
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	// 释放模型资源
	if sdCtx != nil {
		log.Println("Freeing stable-diffusion context...")
		sdCtx.Free()
	}

	log.Println("Server exiting")
}
