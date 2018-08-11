import os
from copy import deepcopy

import h5py
import numpy as np
import operator
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from zeebeez3.transforms.biosound import BiosoundTransform
from zeebeez3.core.utils import DECODER_CALL_TYPES, CALL_TYPE_COLORS, USED_ACOUSTIC_PROPS, ACOUSTIC_FUND_PROPS, \
    decode_if_bytes


class AggregateBiosounds(object):

    def __init__(self):

        self.df = None
        self.data = None
        self.Xraw = None
        self.Xwhitened = None
        self.Xix = None
        self.acoustic_props = None
        self.pca_props = None
        self.pca = None

    def transform(self, birds, data_dir='/auto/tdrive/mschachter/data'):

        acoustic_props = USED_ACOUSTIC_PROPS

        self.data = {'bird':list(), 'stim_id':list(), 'stim_type':list(), 'syllable_order':list(),
                     'start_time':list(), 'end_time':list(), 'xindex':list()}

        self.Xraw = list()

        for bird in birds:
            bs_file = os.path.join(data_dir, bird, 'transforms', 'BiosoundTransform_%s.h5' % bird)
            bst = BiosoundTransform.load(bs_file)

            for (stim_id,syllable_order),gdf in bst.stim_df.groupby(['stim_id', 'order']):

                assert len(gdf) == 1

                stim_type = gdf.stim_type.values[0]
                start_time = gdf.start_time.values[0]
                end_time = gdf.end_time.values[0]

                aprops = list()
                for aprop in acoustic_props:
                    aprops.append(gdf[aprop].values[0])

                xindex = len(self.Xraw)
                self.Xraw.append(np.array(aprops))
                
                self.data['bird'].append(bird)
                self.data['stim_id'].append(stim_id)
                self.data['stim_type'].append(stim_type)
                self.data['syllable_order'].append(syllable_order)
                self.data['start_time'].append(start_time)
                self.data['end_time'].append(end_time)
                self.data['xindex'].append(xindex)

        self.df = pd.DataFrame(self.data)
        self.Xraw = np.array(self.Xraw)
        self.acoustic_props = acoustic_props

        # train a whitening pca transform on the non-duplicate data
        Xz,good_indices = self.remove_duplicates()
        print('good_indices=',good_indices)
        assert len(good_indices) == len(np.unique(good_indices))

        pca = PCA(whiten=False)
        pca.fit(Xz[good_indices, :])

        for k,evar in enumerate(np.cumsum(pca.explained_variance_ratio_)):
            print('PC %d: %0.2f' % (k, evar))

        # transform the raw features into whitened PCA space
        self.Xwhitened = pca.transform(Xz)

        # sort the samples by call type
        index2ct = {xindex:ct for xindex,ct in zip(self.df.xindex, self.df.stim_type)}
        index_and_ct = [(k,index2ct[k]) for k,gi in enumerate(good_indices)]
        index_and_ct.sort(key=operator.itemgetter(1))
        call_type = np.array([index2ct[k] for k in good_indices])
        re_index = [x[0] for x in index_and_ct]

        rcParams.update({'font.size': 11})
        fig = plt.figure()
        fig.subplots_adjust(top=0.95, bottom=0.10, right=0.95, left=0.05, hspace=0.30, wspace=0.30)
        gs = plt.GridSpec(2, 3)

        ax = plt.subplot(gs[0, 0])
        absmax = np.abs(Xz).max()
        plt.imshow(Xz[good_indices, :][re_index, :], interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, vmin=-4, vmax=4)
        plt.xticks(np.arange(len(self.acoustic_props)), self.acoustic_props, rotation=90)
        plt.colorbar()
        plt.title('Z-scored Xraw')

        ax = plt.subplot(gs[0, 1])
        C = np.corrcoef(Xz.T)
        plt.imshow(C, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, origin='lower', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Xraw Corrcoef')
        plt.xticks(np.arange(len(self.acoustic_props)), self.acoustic_props, rotation=90)
        plt.yticks(np.arange(len(self.acoustic_props)), self.acoustic_props)

        ax = plt.subplot(gs[0, 2], projection='3d')
        for ct in DECODER_CALL_TYPES:
            i = call_type == ct
            clr = CALL_TYPE_COLORS[ct]
            ax.scatter(self.Xwhitened[i, 0], self.Xwhitened[i, 1], self.Xwhitened[i, 2], c=clr)
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_zlim([-4, 4])

        ax = plt.subplot(gs[1, 0])
        absmax = 4.
        plt.imshow(self.Xwhitened[good_indices, :][re_index, :], interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, vmin=-absmax, vmax=absmax)
        plt.colorbar()
        plt.title('Xwhitened')

        ax = plt.subplot(gs[1, 1])
        C = np.corrcoef(self.Xwhitened[good_indices, :].T)
        plt.imshow(C, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, origin='lower', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title('Xwhitened Corrcoef')

        ax = plt.subplot(gs[1, 2])
        absmax = np.abs(pca.components_).max()
        plt.imshow(pca.components_, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic,
                   vmin=-absmax, vmax=absmax, origin='lower')
        plt.xticks(np.arange(len(self.acoustic_props)), self.acoustic_props, rotation=90)
        plt.colorbar()
        plt.title('Components')

        plt.show()

        # save the essential components of the PCA
        pca_props_to_save = ['mean_', 'components_', 'explained_variance_']
        self.pca = pca
        self.pca_props = {pprop:np.array(getattr(self.pca, pprop)) for pprop in pca_props_to_save}

    def remove_duplicates(self, aprops=None, thresh=0.99):

        if aprops is None:
            aprops = [aprop for aprop in self.acoustic_props if aprop not in ['meantime']]

        num_total_samps = self.Xraw.shape[0]
        Xred = np.zeros([num_total_samps, len(aprops)])
        for k,aprop in enumerate(aprops):
            i = self.acoustic_props.index(aprop)
            Xred[:, k] = self.Xraw[:, i]

        # get rid of outliers and syllables where the fundamental can't be estimated
        good_i = np.ones([num_total_samps], dtype='bool')
        for k,aprop in enumerate(aprops):
            x = Xred[:, k]
            if aprop not in ACOUSTIC_FUND_PROPS:
                q1 = np.percentile(x, 1)
                q99 = np.percentile(x, 99)
                good_i &= (x > q1) & (x < q99)
            else:
                good_i &= x > 0

        # hold on to the original indices
        full_indices = np.arange(num_total_samps)[good_i]

        # select the subset of good acoustic features
        Xred = Xred[good_i, :]

        # zscore the feature matrix
        Xz = deepcopy(Xred)
        Xz -= Xz.mean(axis=0)
        Xz /= Xz.std(axis=0, ddof=1)
        nsamps, nf = Xz.shape
        print('# of samples used for duplicate detection: %d' % nsamps)

        D = np.zeros([nsamps, nsamps])
        for k in range(nsamps):
            for j in range(nsamps):
                if k == j:
                    D[k, j] = 1.
                    continue
                x = Xz[k, :]
                y = Xz[j, :]
                D[k, j] = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

        # remove the duplicates
        good_indices = list()
        duplicate_indices = list()
        for k in range(nsamps):
            if k in duplicate_indices:
                continue
            good_indices.append(k)
            for j in range(nsamps):
                if j == k:
                    continue
                if j not in duplicate_indices and D[k, j] > thresh:
                    duplicate_indices.append(j)

        """
        print '# of good=%d, # of duplicate=%d' % (len(good_indices), len(duplicate_indices))
        Dgood = np.zeros([len(good_indices), len(good_indices)])
        for k,k1 in enumerate(good_indices):
            for j,k2 in enumerate(good_indices):
                if k == j:
                    Dgood[k, j] = 1.
                x = Xz[k1, :]
                y = Xz[k2, :]
                Dgood[k, j] = np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y))
        """

        return Xz[good_indices, :],full_indices[good_indices]

    def save(self, output_file):

        hf = h5py.File(output_file, 'w')
        hf.attrs['acoustic_props'] = self.acoustic_props
        hf.attrs['col_names'] = list(self.data.keys())
        for cname,cvals in list(self.data.items()):
            hf[cname] = np.array(cvals)

        hf['Xraw'] = self.Xraw
        hf['Xwhitened'] = self.Xwhitened

        pca_grp = hf.create_group('pca')
        for pprop,pval in list(self.pca_props.items()):
            pca_grp[pprop] = np.array(pval)

        hf.close()

    @classmethod
    def load(cls, output_file):
        
        agg = AggregateBiosounds()
        
        hf = h5py.File(output_file, 'r')
        cnames = hf.attrs['col_names']
        
        agg.data = dict()        
        for cname in cnames:
            agg.data[decode_if_bytes(cname)] = np.array(hf[cname])
        agg.df = pd.DataFrame(agg.data)

        agg.acoustic_props = list(hf.attrs['acoustic_props'])
        
        agg.Xraw = np.array(hf['Xraw'])
        agg.Xwhitened = np.array(hf['Xwhitened'])

        agg.pca = PCA(whiten=True)
        pca_grp = hf['pca']
        for key in list(pca_grp.keys()):
            setattr(agg.pca, key, np.array(pca_grp[key]))

        hf.close()

        return agg


if __name__ == '__main__':

    agg_file = '/auto/tdrive/mschachter/data/aggregate/biosound.h5'

    agg = AggregateBiosounds()
    agg.transform(['GreBlu9508M', 'YelBlu6903F', 'WhiWhi4522M'])
    agg.save(agg_file)
    # agg = AggregateBiosounds.load(agg_file)
