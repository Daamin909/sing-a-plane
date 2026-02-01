extends Node2D


func _ready() -> void:
	pass 


func _process(delta: float) -> void:
	pass


func _on_top_boundary_body_entered(body: Node2D) -> void:
	if body is CharacterBody2D:
		get_tree().change_scene_to_file("res://scenes/game_over.tscn") # Replace with function body.

func _on_bottom_boundary_body_entered(body: Node2D) -> void:
	if body is CharacterBody2D:
		get_tree().change_scene_to_file("res://scenes/game_over.tscn") # Replace with function body.
