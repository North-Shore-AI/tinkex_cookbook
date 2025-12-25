# credo:disable-for-this-file Credo.Check.Refactor.CyclomaticComplexity
defmodule TinkexCookbook.Renderers.Vision do
  @moduledoc """
  Image helpers for vision-capable renderers.
  """

  alias HfDatasetsEx.Media.Image, as: HfImage
  alias TinkexCookbook.Types.ImageChunk
  alias Vix.Vips.Image, as: VipsImage
  alias Vix.Vips.Operation

  @type image_processor :: module() | map()

  @spec image_to_chunk(binary() | String.t(), image_processor()) :: ImageChunk.t()
  def image_to_chunk(image_or_str, image_processor) do
    image_bytes = load_image_bytes(image_or_str)
    image = decode_image!(image_bytes)
    image = maybe_convert_to_rgb(image)

    {width, height, _bands} = VipsImage.shape(image)
    image_data = encode_jpeg!(image)

    num_patches = get_number_of_image_patches(image_processor, height, width)
    merge_size = fetch_merge_size(image_processor)
    expected_tokens = div(num_patches, merge_size * merge_size)

    ImageChunk.new(image_data, "jpeg", expected_tokens)
  end

  defp load_image_bytes(image_or_str) when is_binary(image_or_str) do
    if String.valid?(image_or_str) do
      cond do
        String.starts_with?(image_or_str, "data:") ->
          decode_data_uri!(image_or_str)

        String.starts_with?(image_or_str, "http://") or
            String.starts_with?(image_or_str, "https://") ->
          fetch_url!(image_or_str)

        true ->
          raise ArgumentError,
                "The provided image must be a URL, data URI, or binary image data."
      end
    else
      image_or_str
    end
  end

  defp load_image_bytes(other) do
    raise ArgumentError,
          "The provided image must be a URL, data URI, or binary image data: #{inspect(other)}"
  end

  defp decode_data_uri!(uri) do
    case Regex.run(~r/^data:(?<mime>[^;]+);base64,(?<data>.*)$/s, uri) do
      [_full, _mime, data] ->
        case Base.decode64(data) do
          {:ok, bytes} -> bytes
          :error -> raise ArgumentError, "Invalid base64 data in data URI"
        end

      _ ->
        raise ArgumentError, "Invalid data URI format"
    end
  end

  defp fetch_url!(url) do
    case Req.get(url) do
      {:ok, %Req.Response{status: status, body: body}} when status in 200..299 ->
        body

      {:ok, %Req.Response{status: status}} ->
        raise ArgumentError, "Failed to download image: status #{status}"

      {:error, reason} ->
        raise ArgumentError, "Failed to download image: #{inspect(reason)}"
    end
  end

  defp decode_image!(bytes) do
    case HfImage.decode(bytes) do
      {:ok, image} -> image
      {:error, reason} -> raise ArgumentError, "Failed to decode image: #{inspect(reason)}"
    end
  end

  defp maybe_convert_to_rgb(image) do
    {_width, _height, bands} = VipsImage.shape(image)

    if VipsImage.has_alpha?(image) or bands in [2, 4] or bands > 4 do
      case Operation.colourspace(image, :VIPS_INTERPRETATION_RGB) do
        {:ok, converted} ->
          converted

        {:error, reason} ->
          raise ArgumentError, "Failed to convert image to RGB: #{inspect(reason)}"
      end
    else
      image
    end
  end

  defp encode_jpeg!(image) do
    case VipsImage.write_to_buffer(image, ".jpg") do
      {:ok, data} ->
        data

      {:error, reason} ->
        raise ArgumentError, "Failed to encode image as JPEG: #{inspect(reason)}"
    end
  end

  defp get_number_of_image_patches(image_processor, height, width) do
    cond do
      is_atom(image_processor) and
          function_exported?(image_processor, :get_number_of_image_patches, 3) ->
        image_processor.get_number_of_image_patches(height, width, %{})

      is_atom(image_processor) and
          function_exported?(image_processor, :get_number_of_image_patches, 2) ->
        image_processor.get_number_of_image_patches(height, width)

      is_map(image_processor) ->
        fun =
          Map.get(image_processor, :get_number_of_image_patches) ||
            Map.get(image_processor, "get_number_of_image_patches")

        cond do
          is_function(fun, 3) ->
            fun.(height, width, %{})

          is_function(fun, 2) ->
            fun.(height, width)

          true ->
            raise ArgumentError, "Image processor missing get_number_of_image_patches function"
        end

      true ->
        raise ArgumentError, "Invalid image processor: #{inspect(image_processor)}"
    end
  end

  defp fetch_merge_size(image_processor) do
    merge_size =
      cond do
        is_atom(image_processor) and function_exported?(image_processor, :merge_size, 0) ->
          image_processor.merge_size()

        is_map(image_processor) and Map.has_key?(image_processor, :merge_size) ->
          Map.fetch!(image_processor, :merge_size)

        is_map(image_processor) and Map.has_key?(image_processor, "merge_size") ->
          Map.fetch!(image_processor, "merge_size")

        true ->
          raise ArgumentError, "Image processor missing merge_size"
      end

    if not is_integer(merge_size) or merge_size <= 0 do
      raise ArgumentError, "Invalid merge_size: #{inspect(merge_size)}"
    end

    merge_size
  end
end
