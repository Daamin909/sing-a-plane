extends CharacterBody2D

const SPEED = 600.0

func _physics_process(delta: float) -> void:
	var y_dir := Input.get_axis("ui_up", "ui_down")
	velocity.y = y_dir * SPEED

	move_and_slide()
