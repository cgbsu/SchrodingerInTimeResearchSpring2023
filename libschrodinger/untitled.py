def animateImages(
            length : float, 
            images : List[np.array], 
            interval = 1, 
            minimumValue = None, 
            maximumValue = None, 
            lengthRatios = None, 
            potentialRatios = None, 
            baseAlpha : float = .08, 
            colorMap : str = "viridis", 
            showFrame : bool = True, 
            titleFormatString : str = animateImagesDefaultTitleFormatString 
        ): 
    animationFigure = plt.figure()
    animationAxis = animationFigure.add_subplot(xlim=(0, length), ylim=(0, length))
    animationFrame = animationAxis.imshow(
            asNumPyArray(images[0]), 
            extent=[0, length, 0, length], 
            vmin = minimumValue, 
            vmax = maximumValue, 
            zorder = 1, 
            cmap = colorMap
        )
    if showFrame == True: 
        animationAxis.set_title(titleFormatString.format(str(0)))
    elif titleFormatString != None: 
        animationAxis.set_title(titleFormatString)
    if lengthRatios and potentialRatios: 
        constantPotentialRectangles(
                animationAxis, 
                length, 
                lengthRatios, 
                potentialRatios, 
                baseAlpha = baseAlpha
            )
    def animateFrame(frameIndex): 
        if showFrame == True: 
            animationAxis.set_title(titleFormatString.format(str(frameIndex)))
        elif titleFormatString != None: 
            animationAxis.set_title(titleFormatString)
        animationFrame.set_data(asNumPyArray(images[frameIndex]))
        animationFrame.set_zorder(1)
        return animationFrame,
    animation = FuncAnimation(
            animationFigure, 
            animateFrame, 
            interval = interval, 
            frames = np.arange(0, len(images), 2), 
            repeat = True, 
            blit = 0
        )
    return animation


_field_artist = None
def liveplot(sim, component=mp.Ez, vmax=0.1):
    ''' You must put ``mp.at_beginning(liveplot)`` in your arguments to run!
        Make sure to turn the progress_interval up before using this
    '''
    global _field_artist
    field_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=component)
    if sim.meep_time() == 0.:
        ax = sim.plot2D()
        extent = ax.get_images()[0].get_extent()
        _field_artist = plt.imshow(field_data.transpose()[::-1], interpolation='spline36', cmap='RdBu',
                                   alpha=0.8, vmin=-vmax, vmax=vmax, extent=extent)
    else:
        _field_artist.set_data(field_datSSa.transpose())

    plt.title(f't = {sim.meep_time()}')
    display.clear_output(wait=True)
    display.display(plt.gcf())


def x_field_data(sim, component=mp.Ey):
    xspan = sim.cell_size.x/2 * np.linspace(-1, 1, int(sim.cell_size.x * sim.resolution))
    e_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=component)
    return xspan, e_data
