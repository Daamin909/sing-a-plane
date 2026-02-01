extends Node
class_name LRCParser

func parse_lrc(path: String) -> Array:
	var result := []
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Failed to open LRC file")
		return result

	while not file.eof_reached():
		var line := file.get_line().strip_edges()
		if line == "":
			continue

		# Matches [mm:ss.xx]Lyric text
		var regex := RegEx.new()
		regex.compile("\\[(\\d+):(\\d+\\.\\d+)\\](.*)")
		var match := regex.search(line)
		if match:
			var minute := float(match.get_string(1))
			var sec := float(match.get_string(2))
			var time := minute * 60.0 + sec
			var text := match.get_string(3).strip_edges()
			result.append({
				"time": time,
				"text": text
			})

	result.sort_custom(func(a, b): return a.time < b.time)
	return result
