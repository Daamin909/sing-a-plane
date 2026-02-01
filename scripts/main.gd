extends Node2D

@onready var player := $b
@onready var lyric_label := $Label

var lyrics := []
var index := 0




func _ready():
	lyrics = LRCParser.new().parse_lrc("res://assets/lyrics.lrc")
	player.play()
	
	
func _process(_delta):
	if index >= lyrics.size():
		return

	var t = player.get_playback_position()
	if t >= lyrics[index].time:
		lyric_label.text = lyrics[index].text
		index += 1
