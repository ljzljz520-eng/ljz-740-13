package stablediffusion

import (
	"bytes"
	"image/png"
	"strings"
	"testing"
)

// 默认值填充
func TestTxt2ImgNormalize(t *testing.T) {
	p := Txt2ImgParams{Prompt: "a cat"}
	p.normalize()

	if p.Width != DefaultTxt2ImgWidth {
		t.Errorf("默认 width = %d, 期望 %d", p.Width, DefaultTxt2ImgWidth)
	}
	if p.Height != DefaultTxt2ImgHeight {
		t.Errorf("默认 height = %d, 期望 %d", p.Height, DefaultTxt2ImgHeight)
	}
	if p.Steps != DefaultTxt2ImgSteps {
		t.Errorf("默认 steps = %d, 期望 %d", p.Steps, DefaultTxt2ImgSteps)
	}
	if p.CfgScale != DefaultTxt2ImgCfgScale {
		t.Errorf("默认 cfg_scale = %v, 期望 %v", p.CfgScale, DefaultTxt2ImgCfgScale)
	}
	if p.BatchCount != DefaultTxt2ImgBatchCount {
		t.Errorf("默认 batch_count = %d, 期望 %d", p.BatchCount, DefaultTxt2ImgBatchCount)
	}
	if p.Sampler != DefaultTxt2ImgSampler {
		t.Errorf("默认 sampler = %q, 期望 %q", p.Sampler, DefaultTxt2ImgSampler)
	}
	if p.Scheduler != DefaultTxt2ImgScheduler {
		t.Errorf("默认 scheduler = %q, 期望 %q", p.Scheduler, DefaultTxt2ImgScheduler)
	}
}

// 合法参数（含显式值、大小写/空格归一化、seed=0）应通过校验
func TestTxt2ImgValidateOK(t *testing.T) {
	cases := []Txt2ImgParams{
		{Prompt: "a cat"}, // 零值由 normalize 填充
		{Prompt: "  dog  ", Width: 1024, Height: 768, Steps: 30, CfgScale: 5, BatchCount: 4,
			Sampler: "EULER", Scheduler: " Karras ", Seed: 0},
		{Prompt: "x", Width: 64, Height: 4096, Steps: 1, CfgScale: 1.0, BatchCount: 1,
			Sampler: "dpm++2mv2", Scheduler: "bong_tangent"},
		{Prompt: "x", Width: 4096, Height: 64, Steps: 150, CfgScale: 30.0, BatchCount: 16},
	}
	for i, p := range cases {
		p.normalize()
		if err := p.validate(); err != nil {
			t.Errorf("用例 %d 期望通过校验，实际报错: %v", i, err)
		}
	}
}

// 各类非法参数必须在 Go 层报错
func TestTxt2ImgValidateInvalid(t *testing.T) {
	cases := []struct {
		name   string
		params Txt2ImgParams
		want   string // 错误信息中应包含的片段
	}{
		{"空 prompt", Txt2ImgParams{Prompt: "   "}, "prompt 不能为空"},
		{"width 过小", Txt2ImgParams{Prompt: "x", Width: 32, Height: 512}, "width 必须在"},
		{"height 过大", Txt2ImgParams{Prompt: "x", Width: 512, Height: 5000}, "height 必须在"},
		{"width 非 8 倍数", Txt2ImgParams{Prompt: "x", Width: 500, Height: 512}, "width 必须是 8 的倍数"},
		{"steps 为 0 时走默认值", Txt2ImgParams{Prompt: "x", Steps: 0}, ""},
		{"steps 过大", Txt2ImgParams{Prompt: "x", Steps: 999}, "steps 必须在"},
		{"cfg 过小", Txt2ImgParams{Prompt: "x", CfgScale: 0.5}, "cfg_scale 必须在"},
		{"cfg 过大", Txt2ImgParams{Prompt: "x", CfgScale: 100}, "cfg_scale 必须在"},
		{"batch 为 0", Txt2ImgParams{Prompt: "x", BatchCount: 0}, ""},
		{"batch 过大", Txt2ImgParams{Prompt: "x", BatchCount: 100}, "batch_count 必须在"},
		{"非法采样器", Txt2ImgParams{Prompt: "x", Sampler: "not_a_sampler"}, "不支持的采样器"},
		{"非法调度器", Txt2ImgParams{Prompt: "x", Scheduler: "not_a_scheduler"}, "不支持的调度器"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.params.normalize()
			err := tc.params.validate()
			if tc.want == "" {
				if err != nil {
					t.Errorf("期望通过校验，实际: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("期望报错（含 %q），实际通过", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("错误信息 %q 中不包含 %q", err.Error(), tc.want)
			}
		})
	}
}

// 多个非法字段应被一次性聚合返回
func TestTxt2ImgValidateAggregatesErrors(t *testing.T) {
	p := Txt2ImgParams{
		Prompt:     "",
		Width:      10,
		Height:     11,
		Steps:      -1,
		CfgScale:   -1,
		BatchCount: 0,
		Sampler:    "bad",
		Scheduler:  "bad",
	}
	err := p.validate()
	if err == nil {
		t.Fatal("期望聚合错误，实际通过")
	}
	msg := err.Error()
	for _, frag := range []string{
		"prompt 不能为空",
		"width 必须在",
		"height 必须在",
		"steps 必须在",
		"cfg_scale 必须在",
		"不支持的采样器",
		"不支持的调度器",
	} {
		if !strings.Contains(msg, frag) {
			t.Errorf("聚合错误中缺少 %q，完整错误: %s", frag, msg)
		}
	}
}

// 非法参数时即便上下文为 nil 也应先返回参数错误（参数校验优先于后端调用）
func TestTxt2ImgInvalidParamsBeforeBackend(t *testing.T) {
	var c *Context
	_, err := c.Txt2Img(Txt2ImgParams{Prompt: ""})
	if err == nil || !strings.Contains(err.Error(), "文生图参数非法") {
		t.Fatalf("期望参数非法错误，实际: %v", err)
	}
}

// 上下文未初始化时应在 Go 层报错
func TestTxt2ImgNilContext(t *testing.T) {
	var c *Context
	_, err := c.Txt2Img(Txt2ImgParams{Prompt: "a cat"})
	if err == nil || !strings.Contains(err.Error(), "上下文未初始化") {
		t.Fatalf("期望上下文未初始化错误，实际: %v", err)
	}
}

// 随机 seed 生成应为非负且两次不同
func TestNewRandomSeed(t *testing.T) {
	s1, err := newRandomSeed()
	if err != nil {
		t.Fatalf("生成 seed 失败: %v", err)
	}
	s2, err := newRandomSeed()
	if err != nil {
		t.Fatalf("生成 seed 失败: %v", err)
	}
	if s1 < 0 || s2 < 0 {
		t.Fatalf("seed 必须非负: %d, %d", s1, s2)
	}
	if s1 == s2 {
		t.Fatalf("两次随机 seed 相同: %d", s1)
	}
}

// 支持列表应与底层映射表一致
func TestSamplerAndSchedulerMaps(t *testing.T) {
	for _, name := range SupportedSamplers() {
		if _, ok := txt2ImgSamplerNames[name]; !ok {
			t.Errorf("采样器 %q 在支持列表中但映射表缺失", name)
		}
	}
	if len(SupportedSamplers()) != len(txt2ImgSamplerNames) {
		t.Errorf("采样器支持列表(%d)与映射表(%d)数量不一致",
			len(SupportedSamplers()), len(txt2ImgSamplerNames))
	}
	for _, name := range SupportedSchedulers() {
		if _, ok := txt2ImgSchedulerNames[name]; !ok {
			t.Errorf("调度器 %q 在支持列表中但映射表缺失", name)
		}
	}
	if len(SupportedSchedulers()) != len(txt2ImgSchedulerNames) {
		t.Errorf("调度器支持列表(%d)与映射表(%d)数量不一致",
			len(SupportedSchedulers()), len(txt2ImgSchedulerNames))
	}
}

// PNG 编码：RGB / RGBA / 空数据 / 数据残缺 / 非法通道
func TestEncodeImagePNG(t *testing.T) {
	t.Run("RGB", func(t *testing.T) {
		img := &Image{Width: 4, Height: 4, Channel: 3, Data: make([]byte, 4*4*3)}
		data, err := encodeImagePNG(img)
		if err != nil {
			t.Fatalf("编码失败: %v", err)
		}
		cfg, err := png.DecodeConfig(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("PNG 解码失败: %v", err)
		}
		if cfg.Width != 4 || cfg.Height != 4 {
			t.Errorf("PNG 尺寸 = %dx%d，期望 4x4", cfg.Width, cfg.Height)
		}
	})

	t.Run("RGBA", func(t *testing.T) {
		img := &Image{Width: 8, Height: 8, Channel: 4, Data: make([]byte, 8*8*4)}
		data, err := encodeImagePNG(img)
		if err != nil {
			t.Fatalf("编码失败: %v", err)
		}
		if len(data) == 0 || !bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n")) {
			t.Error("输出不是合法 PNG 字节")
		}
	})

	t.Run("空图像", func(t *testing.T) {
		if _, err := encodeImagePNG(&Image{}); err == nil {
			t.Error("期望空图像报错")
		}
	})

	t.Run("数据残缺", func(t *testing.T) {
		img := &Image{Width: 8, Height: 8, Channel: 3, Data: make([]byte, 10)}
		if _, err := encodeImagePNG(img); err == nil {
			t.Error("期望数据残缺报错")
		}
	})

	t.Run("非法通道", func(t *testing.T) {
		img := &Image{Width: 4, Height: 4, Channel: 2, Data: make([]byte, 4*4*2)}
		if _, err := encodeImagePNG(img); err == nil {
			t.Error("期望通道数非法报错")
		}
	})
}
