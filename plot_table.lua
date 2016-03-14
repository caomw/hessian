require 'gnuplot'
require 'paths'

local args = lapp[[
    -i,--input (string)    The data you want to plot
    -n,--name (string)     File name to save the image
    -s,--save (default 'img') Folder to save
    -h,--help (default true) show help document
]]

--os.exit(1)

--for i=1,1000 do 
--    if i == 100 then
--        print(i)
--        os.exit()
--    end
--end

--if(args.help) then print(args); os.exit() end


local filename  =  paths.concat(args.save, args.name)

gnuplot.pngfigure(filename)
-- Creates a figure directly on the png file given with args.name. 
-- This uses Gnuplot terminal png, or pngcairo if available.

a = torch.load(args.input)

gnuplot.plot({
   torch.Tensor(a), '+'
})

gnuplot.plotflush()
