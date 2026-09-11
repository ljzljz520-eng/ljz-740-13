package stablediffusion

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/png"
	"math"
	"strings"

	"github.com/example/stablediffusion/bindings"
)

// 文生图参数的取值范围与默认值
const (
	DefaultTxt2ImgWidth      = 512
	DefaultTxt2ImgHeight     = 512
	DefaultTxt2ImgSteps      = 20
	DefaultTxt2ImgCfgScale   = 7.5
	DefaultTxt2ImgBatchCount = 1
	DefaultTxt2ImgSampler    = "euler_a"
	DefaultTxt2ImgScheduler  = "karras"

	minTxt2ImgDimension = 64
	maxTxt2ImgDimension = 4096
	minTxt2ImgSteps     = 1
	maxTxt2ImgSteps     = 150
	minTxt2ImgCfgScale  = 1.0
	maxTxt2ImgCfgScale  = 30.0
	minTxt2ImgBatch     = 1
	maxTxt2ImgBatch     = 16
	maxPromptLength     = 10000
)

// Txt2ImgParams 是文生图接口的结构体参数。
//
// 字段同时带有 json tag，可直接作为 HTTP 请求体使用。零值字段会在调用时
// 被填充为默认值；显式传入非法值时会在 Go 层直接返回错误，不会进入底层 C 调用。
type Txt2ImgParams struct {
	// Prompt 正向提示词，必填
	Prompt string `json:"prompt"`

	// NegativePrompt 反向提示词，可为空
	NegativePrompt string `json:"negative_prompt,omitempty"`

	// Width / Height 输出图像宽高（像素），必须是 8 的倍数，范围 [64, 4096]，零值默认 512
	Width  int `json:"width,omitempty"`
	Height int `json:"height,omitempty"`

	// Steps 采样步数，范围 [1, 150]，零值默认 20
	Steps int `json:"steps,omitempty"`

	// Seed 随机种子；小于 0（如 -1）表示由 Go 层随机生成一个，
	// 实际使用的 seed 会在返回结果 Txt2ImgOutput.Seed 中给出。0 是合法且可复现的种子。
	Seed int64 `json:"seed,omitempty"`

	// CfgScale Classifier-Free Guidance 强度，范围 [1.0, 30.0]，零值默认 7.5
	CfgScale float32 `json:"cfg_scale,omitempty"`

	// Sampler 采样器名称（如 "euler_a"、"dpm++2m"），空值默认 "euler_a"。
	// 可用列表见 SupportedSamplers。
	Sampler string `json:"sampler,omitempty"`

	// Scheduler 噪声调度器名称（如 "karras"、"discrete"），空值默认 "karras"。
	// 可用列表见 SupportedSchedulers。
	Scheduler string `json:"scheduler,omitempty"`

	// BatchCount 一次生成的输出图像数量，范围 [1, 16]，零值默认 1
	BatchCount int `json:"batch_count,omitempty"`
}

// Txt2ImgImage 是单张生成图像的结果。
type Txt2ImgImage struct {
	// Data PNG 编码后的图像字节，可直接写入文件或作为 HTTP 响应体
	Data []byte `json:"-"`

	// Seed 生成该批次时实际使用的随机种子
	Seed int64 `json:"seed"`

	// Width / Height 图像实际宽高
	Width  uint32 `json:"width"`
	Height uint32 `json:"height"`

	// Format 图像编码格式，当前固定为 "png"
	Format string `json:"format"`
}

// Txt2ImgOutput 是文生图接口的返回结果。
type Txt2ImgOutput struct {
	// Images 生成的图像列表（PNG 字节 + seed）
	Images []*Txt2ImgImage

	// Seed 本次生成实际使用的基准种子；请求传入负数 seed 时为随机生成后的值
	Seed int64

	// Params 填充默认值并规范化之后的参数，便于回显
	Params Txt2ImgParams
}

// txt2ImgSamplerNames 采样器名称 -> 底层枚举。
// 名称与 stable-diffusion.cpp 的 sample_method_to_str 保持一致。
var txt2ImgSamplerNames = map[string]bindings.SampleMethod{
	"euler":         bindings.EULER_SAMPLE_METHOD,
	"euler_a":       bindings.EULER_A_SAMPLE_METHOD,
	"heun":          bindings.HEUN_SAMPLE_METHOD,
	"dpm2":          bindings.DPM2_SAMPLE_METHOD,
	"dpm++2s_a":     bindings.DPMPP2S_A_SAMPLE_METHOD,
	"dpm++2m":       bindings.DPMPP2M_SAMPLE_METHOD,
	"dpm++2mv2":     bindings.DPMPP2Mv2_SAMPLE_METHOD,
	"ipndm":         bindings.IPNDM_SAMPLE_METHOD,
	"ipndm_v":       bindings.IPNDM_V_SAMPLE_METHOD,
	"lcm":           bindings.LCM_SAMPLE_METHOD,
	"ddim_trailing": bindings.DDIM_TRAILING_SAMPLE_METHOD,
	"tcd":           bindings.TCD_SAMPLE_METHOD,
	"res_multistep": bindings.RES_MULTISTEP_SAMPLE_METHOD,
	"res_2s":        bindings.RES_2S_SAMPLE_METHOD,
}

// txt2ImgSchedulerNames 调度器名称 -> 底层枚举。
// 名称与 stable-diffusion.cpp 的 scheduler_to_str 保持一致。
var txt2ImgSchedulerNames = map[string]bindings.Scheduler{
	"discrete":     bindings.DISCRETE_SCHEDULER,
	"karras":       bindings.KARRAS_SCHEDULER,
	"exponential":  bindings.EXPONENTIAL_SCHEDULER,
	"ays":          bindings.AYS_SCHEDULER,
	"gits":         bindings.GITS_SCHEDULER,
	"sgm_uniform":  bindings.SGM_UNIFORM_SCHEDULER,
	"simple":       bindings.SIMPLE_SCHEDULER,
	"smoothstep":   bindings.SMOOTHSTEP_SCHEDULER,
	"kl_optimal":   bindings.KL_OPTIMAL_SCHEDULER,
	"lcm":          bindings.LCM_SCHEDULER,
	"bong_tangent": bindings.BONG_TANGENT_SCHEDULER,
}

// SupportedSamplers 返回文生图接口支持的采样器名称。
func SupportedSamplers() []string {
	return []string{
		"euler", "euler_a", "heun", "dpm2",
		"dpm++2s_a", "dpm++2m", "dpm++2mv2",
		"ipndm", "ipndm_v", "lcm",
		"ddim_trailing", "tcd", "res_multistep", "res_2s",
	}
}

// SupportedSchedulers 返回文生图接口支持的调度器名称。
func SupportedSchedulers() []string {
	return []string{
		"discrete", "karras", "exponential", "ays", "gits",
		"sgm_uniform", "simple", "smoothstep", "kl_optimal",
		"lcm", "bong_tangent",
	}
}

// normalize 填充默认值并规范化文本字段（trim、小写化采样器名称）。
func (p *Txt2ImgParams) normalize() {
	p.Prompt = strings.TrimSpace(p.Prompt)
	p.NegativePrompt = strings.TrimSpace(p.NegativePrompt)

	if p.Width == 0 {
		p.Width = DefaultTxt2ImgWidth
	}
	if p.Height == 0 {
		p.Height = DefaultTxt2ImgHeight
	}
	if p.Steps == 0 {
		p.Steps = DefaultTxt2ImgSteps
	}
	if p.CfgScale == 0 {
		p.CfgScale = DefaultTxt2ImgCfgScale
	}
	if p.BatchCount == 0 {
		p.BatchCount = DefaultTxt2ImgBatchCount
	}

	p.Sampler = strings.ToLower(strings.TrimSpace(p.Sampler))
	if p.Sampler == "" {
		p.Sampler = DefaultTxt2ImgSampler
	}
	p.Scheduler = strings.ToLower(strings.TrimSpace(p.Scheduler))
	if p.Scheduler == "" {
		p.Scheduler = DefaultTxt2ImgScheduler
	}
}

// validate 在 Go 层校验所有参数，收集全部错误后一次性返回，
// 避免把非法参数传给底层 C 接口。
func (p *Txt2ImgParams) validate() error {
	var errs []error

	if p.Prompt == "" {
		errs = append(errs, errors.New("prompt 不能为空"))
	} else if len(p.Prompt) > maxPromptLength {
		errs = append(errs, fmt.Errorf("prompt 长度不能超过 %d 个字符，当前 %d", maxPromptLength, len(p.Prompt)))
	}

	if err := validateDimension("width", p.Width); err != nil {
		errs = append(errs, err)
	}
	if err := validateDimension("height", p.Height); err != nil {
		errs = append(errs, err)
	}

	if p.Steps < minTxt2ImgSteps || p.Steps > maxTxt2ImgSteps {
		errs = append(errs, fmt.Errorf("steps 必须在 [%d, %d] 之间，当前 %d",
			minTxt2ImgSteps, maxTxt2ImgSteps, p.Steps))
	}

	if p.CfgScale != p.CfgScale { // NaN 判断
		errs = append(errs, errors.New("cfg_scale 不能为 NaN"))
	} else if p.CfgScale < minTxt2ImgCfgScale || p.CfgScale > maxTxt2ImgCfgScale {
		errs = append(errs, fmt.Errorf("cfg_scale 必须在 [%.1f, %.1f] 之间，当前 %g",
			minTxt2ImgCfgScale, maxTxt2ImgCfgScale, p.CfgScale))
	}

	if p.BatchCount < minTxt2ImgBatch || p.BatchCount > maxTxt2ImgBatch {
		errs = append(errs, fmt.Errorf("batch_count 必须在 [%d, %d] 之间，当前 %d",
			minTxt2ImgBatch, maxTxt2ImgBatch, p.BatchCount))
	}

	if _, ok := txt2ImgSamplerNames[p.Sampler]; !ok {
		errs = append(errs, fmt.Errorf("不支持的采样器 %q，可选值: %v", p.Sampler, SupportedSamplers()))
	}
	if _, ok := txt2ImgSchedulerNames[p.Scheduler]; !ok {
		errs = append(errs, fmt.Errorf("不支持的调度器 %q，可选值: %v", p.Scheduler, SupportedSchedulers()))
	}

	return errors.Join(errs...)
}

// validateDimension 校验宽高：范围合法且为 8 的倍数（VAE 下采样要求）。
func validateDimension(name string, v int) error {
	if v < minTxt2ImgDimension || v > maxTxt2ImgDimension {
		return fmt.Errorf("%s 必须在 [%d, %d] 之间，当前 %d",
			name, minTxt2ImgDimension, maxTxt2ImgDimension, v)
	}
	if v%8 != 0 {
		return fmt.Errorf("%s 必须是 8 的倍数，当前 %d", name, v)
	}
	return nil
}

// newRandomSeed 使用密码学随机源生成非负 int64 种子。
func newRandomSeed() (int64, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return 0, fmt.Errorf("生成随机 seed 失败: %w", err)
	}
	return int64(binary.LittleEndian.Uint64(b[:]) & math.MaxInt64), nil
}

// Txt2Img 执行文生图。
//
// 调用流程：
//  1. 在 Go 层填充默认值并校验参数，非法时直接返回错误；
//  2. seed < 0 时在 Go 层生成随机种子（保证可从返回值拿到实际 seed）；
//  3. 调用底层生成接口；
//  4. 将原始像素编码为 PNG，返回图像字节与 seed 信息。
func (c *Context) Txt2Img(params Txt2ImgParams) (*Txt2ImgOutput, error) {
	// 1. 默认值 + 校验，全部在 Go 层完成；参数非法时直接报错，不进入底层 C 调用
	params.normalize()
	if err := params.validate(); err != nil {
		return nil, fmt.Errorf("文生图参数非法: %w", err)
	}

	if c == nil || c.ctx == nil {
		return nil, errors.New("stable-diffusion 上下文未初始化，请先调用 NewContext")
	}

	// 2. 随机种子在 Go 层生成，便于回传
	if params.Seed < 0 {
		seed, err := newRandomSeed()
		if err != nil {
			return nil, err
		}
		params.Seed = seed
	}

	method := txt2ImgSamplerNames[params.Sampler]
	scheduler := txt2ImgSchedulerNames[params.Scheduler]

	// 3. 映射到通用生成配置
	cfg := GenerationConfig{
		Prompt:         params.Prompt,
		NegativePrompt: params.NegativePrompt,
		Width:          params.Width,
		Height:         params.Height,
		Seed:           params.Seed,
		BatchCount:     params.BatchCount,
		Sampler: SamplerConfig{
			Scheduler: scheduler,
			Method:    method,
			Steps:     params.Steps,
			TxtCfg:    params.CfgScale,
			ImgCfg:    params.CfgScale,
		},
	}

	images, err := c.GenerateImage(cfg)
	if err != nil {
		return nil, fmt.Errorf("文生图生成失败: %w", err)
	}
	if len(images) == 0 {
		return nil, errors.New("文生图生成失败: 底层未返回任何图像")
	}

	// 4. 编码 PNG 并组装 seed 信息
	out := &Txt2ImgOutput{
		Images: make([]*Txt2ImgImage, 0, len(images)),
		Seed:   params.Seed,
		Params: params,
	}
	for i, img := range images {
		pngData, encErr := encodeImagePNG(img)
		if encErr != nil {
			return nil, fmt.Errorf("第 %d 张图像 PNG 编码失败: %w", i+1, encErr)
		}
		// 底层使用单一 RNG 流生成整个批次，这里回传批次实际使用的基准 seed
		out.Images = append(out.Images, &Txt2ImgImage{
			Data:   pngData,
			Seed:   params.Seed,
			Width:  img.Width,
			Height: img.Height,
			Format: "png",
		})
	}

	return out, nil
}

// encodeImagePNG 将底层返回的原始像素（RGB/RGBA）编码为 PNG 字节。
func encodeImagePNG(img *Image) ([]byte, error) {
	if img == nil || len(img.Data) == 0 {
		return nil, errors.New("图像数据为空")
	}

	w, h := int(img.Width), int(img.Height)
	expected := w * h * int(img.Channel)
	if len(img.Data) < expected {
		return nil, fmt.Errorf("图像数据长度不完整: 需要 %d 字节，实际 %d", expected, len(img.Data))
	}

	rect := image.Rect(0, 0, w, h)
	rgba := image.NewRGBA(rect)

	switch img.Channel {
	case 3: // RGB
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				src := (y*w + x) * 3
				dst := rgba.PixOffset(x, y)
				rgba.Pix[dst+0] = img.Data[src+0]
				rgba.Pix[dst+1] = img.Data[src+1]
				rgba.Pix[dst+2] = img.Data[src+2]
				rgba.Pix[dst+3] = 255
			}
		}
	case 4: // RGBA
		copy(rgba.Pix, img.Data[:w*h*4])
	default:
		return nil, fmt.Errorf("不支持的图像通道数: %d（仅支持 3/4）", img.Channel)
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, rgba); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
