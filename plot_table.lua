require 'gnuplot'
require 'paths'

local args = lapp[[
    -i,--input1 (string)    The data 1 you want to plot (usually train)
    -j,--input2 (default '')    The data 2 you want to plot (usually test)
    -p,--powerRecord (default 'nil') The record of power iteration call
    -n,--name (string)     File name to save the image
    -s,--save (default 'img') Folder to save
    -e,--epochPlot plot for test/train errors per epoch
    -t,--epochPlotTensor  plot for test/train error/acc that is stored as lua table
    --epochCompareTestAcc plot for comparing test1.csv test2.csv 
    --compareName1 (default 'hoge1') name used as legend for test1.csv 
    --compareName2 (default 'hoge2') name used as legend for test2.csv
    -x,--xlabel (string)  xlabel
    -y,--ylabel (string)  ylabel
]]


--if(args.help) then print(args); os.exit() end


local filename  =  paths.concat(args.save, args.name)

print(args.epochPlotTensor)

if args.epochCompareTestAcc then
    gnuplot.epsfigure(filename)
    -- Creates a figure directly on the png file given with args.name. 
    -- This uses Gnuplot terminal png, or pngcairo if available.

    require 'csvigo' 
    print("ok")
    local test1 = csvigo.load(args.input1)
    test1 = torch.Tensor(test1["test_acc"])
    local test2 = csvigo.load(args.input2)
    test2 = torch.Tensor(test2["test_acc"])

    --power = torch.load(args.powerRecord)


    gnuplot.plot({args.compareName1,test1, "-"},{args.compareName2,test2, "-"})

    gnuplot.plotflush()  

elseif args.epochPlot then
    gnuplot.pngfigure(filename)
    -- Creates a figure directly on the png file given with args.name. 
    -- This uses Gnuplot terminal png, or pngcairo if available.

    require 'csvigo' 
    print("ok")
    local train = csvigo.load(args.input1)
    train = torch.Tensor(train["train_acc"])
    local test = csvigo.load(args.input2)
    test = torch.Tensor(test["test_acc"])

    --power = torch.load(args.powerRecord)


    gnuplot.plot({test, "-"},{train, "-"})

    gnuplot.plotflush()  

elseif args.epochPlotTensor then
    print(filename)
    gnuplot.epsfigure(filename)
    print(args.input1)
    local train = torch.Tensor(torch.load(args.input1))
    print(args.input2)
    local test = torch.Tensor(torch.load(args.input2))

    gnuplot.xlabel(args.xlabel)
    gnuplot.ylabel(args.ylabel)

    gnuplot.plot({"test",test, "-"},{"train", train, "-"})

    gnuplot.plotflush()
else
    gnuplot.pngfigure(filename)
    a = torch.load(args.input1)

    gnuplot.xlabel(args.xlabel)
    gnuplot.ylabel(args.ylabel)

    gnuplot.plot({torch.Tensor(a), '+'})

    gnuplot.plotflush()
end

