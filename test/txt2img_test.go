package test

import (
	"encoding/json"
	"strings"
	"testing"

	sd "github.com/example/stablediffusion"
)

// 从 JSON 请求体解析文生图结构体参数
func TestTxt2ImgParamsJSON(t *testing.T) {
	body := `{
		"prompt": "a cute cat, high quality",
		"negative_prompt": "blurry, low quality",
		"width": 768,
		"height": 512,
		"steps": 25,
		"seed": -1,
		"cfg_scale": 6.5,
		"sampler": "dpm++2m",
		"scheduler": "karras",
		"batch_count": 2
	}`

	var params sd.Txt2ImgParams
	if err := json.Unmarshal([]byte(body), &params); err != nil {
		t.Fatalf("JSON 解析失败: %v", err)
	}
	if params.Prompt == "" || params.Width != 768 || params.CfgScale != 6.5 ||
		params.Sampler != "dpm++2m" || params.BatchCount != 2 {
		t.Fatalf("JSON 字段映射错误: %+v", params)
	}
}

// 非法参数在 Go 层先报错（此时根本没有模型/上下文）
func TestTxt2ImgInvalidParamsRejectedAtGoLayer(t *testing.T) {
	var ctx *sd.Context // nil，模拟未加载模型

	_, err := ctx.Txt2Img(sd.Txt2ImgParams{
		Prompt:     "",
		Width:      100, // 非 8 倍数
		Height:     99999,
		Steps:      0,
		CfgScale:   999,
		BatchCount: 50,
		Sampler:    "fake_sampler",
	})
	if err == nil {
		t.Fatal("非法参数应在 Go 层报错")
	}
	if !strings.Contains(err.Error(), "文生图参数非法") {
		t.Fatalf("错误应来自 Go 层参数校验，实际: %v", err)
	}
	t.Logf("Go 层校验错误: %v", err)
}

// 参数合法但上下文未初始化时，应在通过校验之后报上下文错误
func TestTxt2ImgValidParamsPassValidation(t *testing.T) {
	var ctx *sd.Context

	_, err := ctx.Txt2Img(sd.Txt2ImgParams{
		Prompt:  "a dog on the grass",
		Sampler: "EULER_A", // 大小写不敏感
		Seed:    42,
	})
	if err == nil {
		t.Fatal("无上下文时应报错")
	}
	if strings.Contains(err.Error(), "参数非法") {
		t.Fatalf("合法参数不应被判定为非法: %v", err)
	}
	if !strings.Contains(err.Error(), "上下文未初始化") {
		t.Fatalf("期望上下文未初始化错误，实际: %v", err)
	}
}

// 支持列表可用于给调用方做提示
func TestSupportedSamplers(t *testing.T) {
	if len(sd.SupportedSamplers()) == 0 || len(sd.SupportedSchedulers()) == 0 {
		t.Fatal("采样器/调度器支持列表不应为空")
	}
}
