// Package bf16 provides a zero-dependency Go client for the bf16.ai
// uncensored 1M-context LLM inference API (https://bf16.ai).
//
// bf16.ai serves verified open-weight models (GLM-5.3, DeepSeek V4.1 Flash,
// Kimi K3.1 Pro) behind OpenAI-compatible (/v1/chat/completions, /v1/responses)
// and Anthropic Messages (/v1/messages) endpoints with strict Zero Data
// Retention (ZDR) and automatic prompt caching.
//
// Quick start:
//
//	client := bf16.NewClient(os.Getenv("BF16_API_KEY"))
//	resp, err := client.ChatCompletion(ctx, bf16.ChatCompletionRequest{
//	    Model: bf16.ModelDeepSeekV41FlashFast,
//	    Messages: []bf16.Message{
//	        {Role: "user", Content: "Hello from Go!"},
//	    },
//	})
package bf16

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	// DefaultBaseURL is the OpenAI-compatible bf16.ai gateway endpoint.
	DefaultBaseURL = "https://api.bf16.ai/v1"

	// AnthropicBaseURL is the native Anthropic Messages endpoint (without /v1).
	AnthropicBaseURL = "https://api.bf16.ai"

	// ModelGLM53 is the flagship 1M-context uncensored coding & reasoning model.
	ModelGLM53 = "zai-org/glm-5.3-uncensored"

	// ModelGLM53Flash is the 1M-context vision-capable flash model.
	ModelGLM53Flash = "zai-org/glm-5.3-flash-uncensored"

	// ModelDeepSeekV41Flash is the 1M-context low-cost flash + vision model.
	ModelDeepSeekV41Flash = "deepseek/deepseek-v4.1-flash-uncensored"

	// ModelDeepSeekV41FlashFast is the turbo throughput 1M-context flash model.
	ModelDeepSeekV41FlashFast = "deepseek/deepseek-v4.1-flash-uncensored-fast"

	// ModelKimiK31Pro is the 2.8T-parameter 1M-context reasoning & vision model.
	ModelKimiK31Pro = "moonshotai/kimi-k3.1-pro-uncensored"

	// ModelAuto routes automatically across active bf16.ai models.
	ModelAuto = "bf16/auto"
)

// Client is a lightweight, goroutine-safe client for https://api.bf16.ai/v1.
type Client struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	UserAgent  string
}

// Option configures a Client instance.
type Option func(*Client)

// WithBaseURL overrides the API base URL (defaults to DefaultBaseURL).
func WithBaseURL(baseURL string) Option {
	return func(c *Client) {
		c.BaseURL = strings.TrimRight(baseURL, "/")
	}
}

// WithHTTPClient provides a custom *http.Client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		if hc != nil {
			c.HTTPClient = hc
		}
	}
}

// NewClient creates a new bf16.ai API client.
func NewClient(apiKey string, opts ...Option) *Client {
	c := &Client{
		APIKey:  strings.TrimSpace(apiKey),
		BaseURL: DefaultBaseURL,
		HTTPClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		UserAgent: "bf16-go/0.1.0",
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// APIError represents an error response returned by the bf16.ai gateway.
type APIError struct {
	StatusCode int
	Message    string
	Type       string
	Code       string
}

func (e *APIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("bf16 api error (%d %s): %s", e.StatusCode, e.Code, e.Message)
	}
	return fmt.Sprintf("bf16 api error (%d): %s", e.StatusCode, e.Message)
}

// Message represents a single chat turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest is the payload for POST /v1/chat/completions.
type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Stop        []string  `json:"stop,omitempty"`
}

// Usage reports prompt, cached, and completion token counts.
type Usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
}

// ChatCompletionChoice is one candidate completion.
type ChatCompletionChoice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatCompletionResponse is returned by POST /v1/chat/completions.
type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   *Usage                 `json:"usage,omitempty"`
}

// StreamChunk represents a single SSE delta chunk from streaming completions.
type StreamChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Index int `json:"index"`
		Delta struct {
			Role             string `json:"role,omitempty"`
			Content          string `json:"content,omitempty"`
			ReasoningContent string `json:"reasoning_content,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage *Usage `json:"usage,omitempty"`
}

// ModelInfo describes an active model returned by GET /v1/models.
type ModelInfo struct {
	ID            string `json:"id"`
	Object        string `json:"object"`
	OwnedBy       string `json:"owned_by,omitempty"`
	ContextLength int    `json:"context_length,omitempty"`
}

// ModelsResponse wraps GET /v1/models.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// AccountInfo describes live balance, tier, and concurrency from GET /v1/me.
type AccountInfo struct {
	Balance struct {
		AvailableUSD float64 `json:"available_usd"`
		ReservedUSD  float64 `json:"reserved_usd"`
	} `json:"balance"`
	Tier struct {
		ID                    int    `json:"id"`
		Name                  string `json:"name"`
		MaxConcurrentRequests int    `json:"max_concurrent_requests"`
		MaxInputTokens        int    `json:"max_input_tokens"`
		RequestsPerMinute     int    `json:"requests_per_minute"`
	} `json:"tier"`
	Concurrency struct {
		Limit int `json:"limit"`
		InUse int `json:"in_use"`
	} `json:"concurrency"`
}

// ListModels fetches the live bf16.ai model catalog from GET /v1/models.
// No API key is required for public catalog reads.
func (c *Client) ListModels(ctx context.Context) (*ModelsResponse, error) {
	var out ModelsResponse
	if err := c.doJSON(ctx, http.MethodGet, "/models", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// Me fetches the authenticated account's live USD balance, tier, and
// concurrency limits from GET /v1/me.
func (c *Client) Me(ctx context.Context) (*AccountInfo, error) {
	if c.APIKey == "" {
		return nil, errors.New("bf16: API key is required for GET /v1/me")
	}
	var out AccountInfo
	if err := c.doJSON(ctx, http.MethodGet, "/me", nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ChatCompletion executes a non-streaming POST /v1/chat/completions request.
func (c *Client) ChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	if c.APIKey == "" {
		return nil, errors.New("bf16: API key is required")
	}
	if req.Model == "" {
		req.Model = ModelDeepSeekV41FlashFast
	}
	req.Stream = false

	var out ChatCompletionResponse
	if err := c.doJSON(ctx, http.MethodPost, "/chat/completions", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// ChatCompletionStream executes a streaming POST /v1/chat/completions request,
// invoking onChunk for each parsed Server-Sent Event until [DONE] or error.
func (c *Client) ChatCompletionStream(
	ctx context.Context,
	req ChatCompletionRequest,
	onChunk func(StreamChunk) error,
) error {
	if c.APIKey == "" {
		return errors.New("bf16: API key is required")
	}
	if req.Model == "" {
		req.Model = ModelDeepSeekV41FlashFast
	}
	req.Stream = true

	payload, err := json.Marshal(req)
	if err != nil {
		return err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return err
	}
	c.applyHeaders(httpReq)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return parseAPIError(resp)
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if err := onChunk(chunk); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func (c *Client) doJSON(ctx context.Context, method, path string, body any, dst any) error {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(raw)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, reader)
	if err != nil {
		return err
	}
	c.applyHeaders(req)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return parseAPIError(resp)
	}
	return json.NewDecoder(resp.Body).Decode(dst)
}

func (c *Client) applyHeaders(req *http.Request) {
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	if c.UserAgent != "" {
		req.Header.Set("User-Agent", c.UserAgent)
	}
}

func parseAPIError(resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
	var parsed struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &parsed); err == nil && parsed.Error.Message != "" {
		return &APIError{
			StatusCode: resp.StatusCode,
			Message:    parsed.Error.Message,
			Type:       parsed.Error.Type,
			Code:       parsed.Error.Code,
		}
	}
	msg := strings.TrimSpace(string(raw))
	if msg == "" {
		msg = http.StatusText(resp.StatusCode)
	}
	return &APIError{
		StatusCode: resp.StatusCode,
		Message:    msg,
	}
}
