defmodule TinkexCookbook.Utils.TinkexConvert do
  @moduledoc """
  Helpers to convert TinkexCookbook types into Tinkex SDK types.
  """

  alias CrucibleTrain.Types, as: CookbookTypes
  alias Tinkex.Types.{EncodedTextChunk, ImageChunk, ModelInput, TensorData}

  @spec model_input_to_tinkex(CookbookTypes.ModelInput.t()) :: ModelInput.t()
  def model_input_to_tinkex(%CookbookTypes.ModelInput{chunks: chunks}) do
    %ModelInput{chunks: Enum.map(chunks, &chunk_to_tinkex/1)}
  end

  @spec datum_to_tinkex(CookbookTypes.Datum.t()) :: map()
  def datum_to_tinkex(%CookbookTypes.Datum{model_input: model_input, loss_fn_inputs: inputs}) do
    %{
      model_input: model_input_to_tinkex(model_input),
      loss_fn_inputs: Map.new(inputs, fn {key, value} -> {key, tensor_data_to_tinkex(value)} end)
    }
  end

  @spec tensor_data_to_tinkex(CookbookTypes.TensorData.t()) :: TensorData.t()
  def tensor_data_to_tinkex(%CookbookTypes.TensorData{} = data) do
    %TensorData{data: data.data, dtype: data.dtype, shape: data.shape}
  end

  defp chunk_to_tinkex(%CookbookTypes.EncodedTextChunk{tokens: tokens}) do
    %EncodedTextChunk{tokens: tokens, type: "encoded_text"}
  end

  defp chunk_to_tinkex(%CookbookTypes.ImageChunk{} = chunk) do
    %ImageChunk{
      data: Base.encode64(chunk.data),
      format: normalize_image_format(chunk.format),
      expected_tokens: chunk.expected_tokens,
      type: "image"
    }
  end

  defp normalize_image_format("png"), do: :png
  defp normalize_image_format("jpeg"), do: :jpeg
  defp normalize_image_format("jpg"), do: :jpeg

  defp normalize_image_format(format) do
    raise ArgumentError, "Unsupported image format: #{inspect(format)}"
  end
end
