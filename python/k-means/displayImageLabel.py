# useful packages
import matplotlib.pyplot as plt


# def displayImageLabel
def displayImageLabel(label, img):
    '''
    fonction qui affiche l'image résultat de classification
    '''

    # locales
    nbLig, nbCol, nbComp = img.shape

    # mise en forme
    imgLabel = label.reshape((nbLig, nbCol))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img)
    axs[0].set_title('Aral sea')

    img1 = axs[1].imshow(imgLabel)
    axs[1].set_title('Classification')

    # options d'affichage
    plt.colorbar(img1, ax=axs[1])

    plt.show()

    return imgLabel
