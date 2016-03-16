require 'gnuplot'
require 'paths'

local args = lapp[[
    -i,--input (string)    The data you want to plot
    -n,--name (string)     File name to save the image
    -s,--save (default 'img') Folder to save
    -h,--help (default true) show help document
    -e,--epochPlot (default false) plot for test/train errors per epoch
]]


--if(args.help) then print(args); os.exit() end


local filename  =  paths.concat(args.save, args.name)

gnuplot.pngfigure(filename)
-- Creates a figure directly on the png file given with args.name. 
-- This uses Gnuplot terminal png, or pngcairo if available.

if args.epochPlot == true then
    

else
    a = torch.load(args.input)

    gnuplot.plot({
        torch.Tensor(a), '+'
    })

    gnuplot.plotflush()
then

