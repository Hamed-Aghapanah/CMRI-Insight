def localizer_draw (slice1 ,all_slices ) :
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    plt.close('all')
    
    page_data = []
    
    
    for ii in range(all_slices):
        i = ii /all_slices
        page_data.append((np.array([[0, 0, i], [1, 0, i], [1, 1, i], [0, 1, i]]), 'gray'))
    
    page_data = np.array(page_data)
    
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot pages
    for page, color in page_data:
        ax.add_collection3d(
            Poly3DCollection([page], color=color, alpha=0.5, edgecolor='black')
        )
    
    # Plot red page
    ax.add_collection3d(
        Poly3DCollection([page_data[slice1][0]], color='red', alpha=0.5)
    )
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('slice '+str(slice1+1) +' / '+ str(  all_slices))
    # Show plot
    plt.show()
    plt.savefig('localizer.png')
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    # Add text to the plot
    ax.text(x=0.5, y=0.5, z=0.5, s='Hamed Aghapanah', fontsize=40, ha='center', va='center')
    
    # plt.pause (0.1)
    plt.close ('all')
    
    import cv2
    image = cv2.imread ('localizer.png')
    
    print('localiz ' ,np.shape (image))
    return image

# if False :
if True :
    a = localizer_draw (20 ,25 )