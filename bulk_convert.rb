require "pp"
require "rmagick"
require_relative "AnimeFace"

if ARGV.size == 0
  warn "Usage: #{$0} <source-dir> <target-dir>"
  exit(-1)
end

Dir.foreach(ARGV[0]) do |item|
  next if item == '.' or item == '..'
  image = Magick::ImageList.new(File.join(ARGV[0], item))
  faces = AnimeFace::detect(image)
  counter = 0
  pp faces
  faces.each do |ctx|
    next if ctx["likelihood"] < 0.8
    counter = counter + 1
    filename = item.split(".").first + "_out_" + counter.to_s + ".jpg"
    output = File.join(ARGV[1], filename)
    face = ctx["face"]
    x = [face["x"] - 30, 0].max
    y = [face["y"] - 30, 0].max
    width = [face["width"] + 60, image.columns - x].min
    height = [face["height"] + 60, image.rows - y].min
    gc = image.crop(x, y, width, height)
    # gc = image.crop(face["x"], face["y"], face["width"], face["height"])
    gc.write(output)
  end
end
