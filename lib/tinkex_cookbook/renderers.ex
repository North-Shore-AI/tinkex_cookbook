defmodule TinkexCookbook.Renderers do
  @moduledoc """
  Renderer registry and lookup helpers.
  """

  alias TinkexCookbook.Renderers.Renderer

  @renderers %{
    "llama3" => TinkexCookbook.Renderers.Llama3,
    "qwen3" => TinkexCookbook.Renderers.Qwen3,
    "qwen3_vl" => TinkexCookbook.Renderers.Qwen3VL,
    "qwen3_vl_instruct" => TinkexCookbook.Renderers.Qwen3VLInstruct,
    "qwen3_disable_thinking" => TinkexCookbook.Renderers.Qwen3DisableThinking,
    "qwen3_instruct" => TinkexCookbook.Renderers.Qwen3Instruct,
    "deepseekv3" => TinkexCookbook.Renderers.DeepSeekV3,
    "deepseekv3_disable_thinking" => TinkexCookbook.Renderers.DeepSeekV3DisableThinking,
    "kimi_k2" => TinkexCookbook.Renderers.KimiK2,
    "gpt_oss_no_sysprompt" => {TinkexCookbook.Renderers.GptOss, [use_system_prompt: false]},
    "gpt_oss_low_reasoning" =>
      {TinkexCookbook.Renderers.GptOss, [use_system_prompt: true, reasoning_effort: "low"]},
    "gpt_oss_medium_reasoning" =>
      {TinkexCookbook.Renderers.GptOss, [use_system_prompt: true, reasoning_effort: "medium"]},
    "gpt_oss_high_reasoning" =>
      {TinkexCookbook.Renderers.GptOss, [use_system_prompt: true, reasoning_effort: "high"]},
    "role_colon" => TinkexCookbook.Renderers.RoleColon
  }

  @spec get(String.t(), map(), keyword()) :: {:ok, Renderer.state()} | {:error, term()}
  def get(name, tokenizer, opts \\ []) do
    case Map.fetch(@renderers, name) do
      {:ok, {module, extra_opts}} ->
        module.init([{:tokenizer, tokenizer} | extra_opts] ++ opts)

      {:ok, module} ->
        module.init([{:tokenizer, tokenizer} | opts])

      :error ->
        {:error, {:unknown_renderer, name}}
    end
  end

  @spec lookup(String.t()) :: {:ok, module(), keyword()} | {:error, term()}
  def lookup(name) do
    case Map.fetch(@renderers, name) do
      {:ok, {module, extra_opts}} -> {:ok, module, extra_opts}
      {:ok, module} -> {:ok, module, []}
      :error -> {:error, {:unknown_renderer, name}}
    end
  end

  @spec supported_renderers() :: [String.t()]
  def supported_renderers, do: Map.keys(@renderers)
end
