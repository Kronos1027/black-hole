# SPDX-FileCopyrightText: 2026 Darlan Pereira da Silva (Kronos1027)
# SPDX-License-Identifier: MIT
# Author: Darlan Pereira da Silva <darlan1027pc@gmail.com>
# Project: Black Hole (BLKH) — https://github.com/Kronos1027/black-hole
#
# BLKH Texture Loader for Godot 4 (GDScript)
#
# Usage:
#   var loader = BLKHTextureLoader.new()
#   loader.server_url = "http://localhost:8080"
#   var texture = await loader.load_texture("skybox.blkh8")
#   $Sprite2D.texture = texture

extends Node
class_name BLKHTextureLoader

## URL of the BLKH Texture Streaming Server
@export var server_url: String = "http://localhost:8080"

## If true, server decodes to PNG (simple). If false, raw BLKH recipe (needs native decoder).
@export var server_side_decode: bool = true

## Load a BLKH texture by name. Returns ImageTexture or null on error.
func load_texture(texture_name: String) -> ImageTexture:
	var url: String
	if server_side_decode:
		url = "%s/texture/%s/decode" % [server_url, texture_name]
	else:
		url = "%s/texture/%s" % [server_url, texture_name]
	
	var http = HTTPRequest.new()
	add_child(http)
	
	var err = http.request(url)
	if err != OK:
		printerr("[BLKH] Failed to request '%s': error %d" % [texture_name, err])
		http.queue_free()
		return null
	
	var response = await http.request_completed
	http.queue_free()
	
	var result = response[0]
	var response_code = response[1]
	var body = response[3]
	
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		printerr("[BLKH] Failed to load '%s': HTTP %d" % [texture_name, response_code])
		return null
	
	# Parse PNG bytes into Image
	var image = Image.new()
	var img_err = image.load_png_from_buffer(body)
	if img_err != OK:
		printerr("[BLKH] Failed to parse PNG for '%s'" % texture_name)
		return null
	
	var texture = ImageTexture.create_from_image(image)
	print("[BLKH] Loaded texture '%s': %dx%d" % [texture_name, texture.get_width(), texture.get_height()])
	return texture

## Load multiple textures in parallel. Returns Array of ImageTexture.
func load_textures(texture_names: Array) -> Array:
	var textures: Array = []
	textures.resize(texture_names.size())
	var loaded_count = 0
	
	for i in range(texture_names.size()):
		_load_texture_async(texture_names[i], func(texture, index):
			textures[index] = texture
			loaded_count += 1
		, i)
	
	# Wait for all to complete
	while loaded_count < texture_names.size():
		await get_tree().process_frame
	
	return textures

func _load_texture_async(texture_name: String, callback: Callable, index: int) -> void:
	var texture = await load_texture(texture_name)
	callback.call(texture, index)

## Get metadata about a texture without downloading it.
func get_texture_info(texture_name: String) -> Dictionary:
	var url = "%s/texture/%s/info" % [server_url, texture_name]
	var http = HTTPRequest.new()
	add_child(http)
	
	http.request(url)
	var response = await http.request_completed
	http.queue_free()
	
	var result = response[0]
	var response_code = response[1]
	var body = response[3]
	
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		return {}
	
	var json = JSON.new()
	if json.parse(body.get_string_from_utf8()) == OK:
		return json.data
	return {}

## List all available textures on the server.
func list_textures() -> Dictionary:
	var url = "%s/textures" % server_url
	var http = HTTPRequest.new()
	add_child(http)
	
	http.request(url)
	var response = await http.request_completed
	http.queue_free()
	
	var result = response[0]
	var body = response[3]
	
	if result != HTTPRequest.RESULT_SUCCESS:
		return {}
	
	var json = JSON.new()
	if json.parse(body.get_string_from_utf8()) == OK:
		return json.data
	return {}

## Compress a PNG file on the server and save as BLKH.
func compress_texture(png_path: String, output_name: String) -> Dictionary:
	var url = "%s/compress?name=%s" % [server_url, output_name]
	var file = FileAccess.open(png_path, FileAccess.READ)
	if not file:
		printerr("[BLKH] Cannot read file: %s" % png_path)
		return {}
	var png_data = file.get_buffer(file.get_length())
	file.close()
	
	var http = HTTPRequest.new()
	add_child(http)
	
	# Godot 4 HTTPRequest doesn't support POST body directly,
	# so we use a custom approach
	var headers = ["Content-Type: image/png"]
	http.request(url, headers, HTTPClient.METHOD_POST, png_data)
	
	var response = await http.request_completed
	http.queue_free()
	
	var result = response[0]
	var response_code = response[1]
	var body = response[3]
	
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		printerr("[BLKH] Compress failed: HTTP %d" % response_code)
		return {}
	
	var json = JSON.new()
	if json.parse(body.get_string_from_utf8()) == OK:
		return json.data
	return {}


## Example usage (call from _ready in your main scene):
# func _ready():
#     var loader = BLKHTextureLoader.new()
#     loader.server_url = "http://localhost:8080"
#     add_child(loader)
#     var texture = await loader.load_texture("skybox.blkh8")
#     if texture:
#         $Sprite2D.texture = texture
