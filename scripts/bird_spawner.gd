extends Node2D

@export var scene: PackedScene
@export var spawn_interval: float = 1.5
@export var min_y: float = -420.0
@export var max_y: float = 420.0
@export var spawn_x: float = 1300.0

@onready var bird_container: Node2D = $"../BirdContainer"

func _ready() -> void:
	assert(bird_container != null)

	var timer := Timer.new()
	timer.wait_time = spawn_interval
	timer.autostart = true
	timer.timeout.connect(_spawn_bird)
	add_child(timer)

func _spawn_bird() -> void:
	var bird: Node2D = scene.instantiate()
	
	var scenes: Array[Node] = bird.get_children()

	for child in bird.get_children():
		if child is CanvasItem:
			scenes.append(child)

	var chosen = scenes.pick_random()

	for n in scenes:
		n.visible = (n == chosen)
	
	bird.position = Vector2(
		spawn_x,
		randf_range(min_y, max_y)
	)

	bird_container.add_child(bird)
