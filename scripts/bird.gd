extends Node2D

@export var speed := 300
@export var despawn_x := -200  # screen-space left

func _process(delta: float) -> void:
	position.x -= speed * delta

	var cam := get_viewport().get_camera_2d()
	if cam == null:
		return

	var screen_left := cam.global_position.x - get_viewport_rect().size.x / 2

	if global_position.x < screen_left + despawn_x:
		queue_free()


func _on_area_2d_body_entered(body: Node2D) -> void:
	if body is CharacterBody2D:
		get_tree().change_scene_to_file("res://scenes/game_over.tscn") # Replace with function body.
