defmodule TinkexCookbook.Types do
  @moduledoc """
  Core types for the Tinkex Cookbook.

  This module contains the fundamental Tinker API types used throughout
  the training pipeline:

  - `TensorData` - Wrapper for tensor data with shape/dtype
  - `ModelInput` - Container for model input chunks
  - `EncodedTextChunk` - Text token chunk
  - `ImageChunk` - Image data chunk
  - `Datum` - Training datum with model input and loss inputs

  For renderer types (Message, TextPart, etc.) see `TinkexCookbook.Renderers.Types`.
  For TrainOnWhat enum see `TinkexCookbook.Renderers.TrainOnWhat`.
  """

  alias TinkexCookbook.Types.{Datum, EncodedTextChunk, ImageChunk, ModelInput, TensorData}

  @type tensor_data :: TensorData.t()
  @type model_input :: ModelInput.t()
  @type encoded_text_chunk :: EncodedTextChunk.t()
  @type image_chunk :: ImageChunk.t()
  @type datum :: Datum.t()
end
