# bf16-go

[![Go Reference](https://pkg.go.dev/badge/github.com/BF16-AI/bf16-go.svg)](https://pkg.go.dev/github.com/BF16-AI/bf16-go)

Official zero-dependency Go SDK for **[bf16.ai](https://bf16.ai)** — uncensored 1M-context LLM inference API with automatic prompt caching and strict Zero Data Retention (ZDR).

## Install

```bash
go get github.com/BF16-AI/bf16-go
```

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	bf16 "github.com/BF16-AI/bf16-go"
)

func main() {
	client := bf16.NewClient(os.Getenv("BF16_API_KEY"))

	// Check live balance & concurrency tier via GET /v1/me
	me, err := client.Me(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Balance: $%.4f (%s)\n", me.Balance.AvailableUSD, me.Tier.Name)

	// Run a 1M-context chat completion
	resp, err := client.ChatCompletion(context.Background(), bf16.ChatCompletionRequest{
		Model: bf16.ModelDeepSeekV41FlashFast,
		Messages: []bf16.Message{
			{Role: "user", Content: "Give 3 practical tips for prompt-cache hit rates in coding agents."},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.Choices[0].Message.Content)
}
```

## Streaming

```go
err := client.ChatCompletionStream(ctx, bf16.ChatCompletionRequest{
	Model:    bf16.ModelGLM53,
	Messages: []bf16.Message{{Role: "user", Content: "Audit this Go middleware for auth bypass risks."}},
}, func(chunk bf16.StreamChunk) error {
	if len(chunk.Choices) > 0 {
		fmt.Print(chunk.Choices[0].Delta.Content)
	}
	return nil
})
```

## Links

- Website & Console (`$0.20` free starter credit): [https://bf16.ai](https://bf16.ai)
- API Documentation: [https://bf16.ai/docs](https://bf16.ai/docs)
- Live Model Catalog & Pricing: [https://bf16.ai/models](https://bf16.ai/models)
- Coding Agent Cost Calculator: [https://bf16.ai/tools/cost-calculator](https://bf16.ai/tools/cost-calculator)
- Official CLI (`npx bf16ai claude`): [https://github.com/BF16-AI/bf16](https://github.com/BF16-AI/bf16)

## License

MIT © BF16 Team
