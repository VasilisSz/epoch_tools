import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import mne
from epoch_tools import Epochs

class TestComparePSD:
    """Comprehensive unit tests for the compare_psd method."""
    
    @pytest.fixture
    def sample_epochs(self):
        """Create a sample Epochs object for testing."""
        # Create mock MNE epochs with realistic metadata
        info = mne.create_info(['C3', 'C4', 'Cz', 'Fz'], 250, 'eeg')
        n_epochs = 100
        n_channels = 4
        n_times = 500  # 2 seconds at 250 Hz
        
        # Create synthetic EEG data
        np.random.seed(42)
        data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
        
        # Add some realistic frequency content
        times = np.arange(n_times) / 250
        for epoch in range(n_epochs):
            for ch in range(n_channels):
                # Add alpha (10 Hz) and theta (6 Hz) components
                data[epoch, ch, :] += 2e-6 * np.sin(2 * np.pi * 10 * times)
                data[epoch, ch, :] += 1e-6 * np.sin(2 * np.pi * 6 * times)
        
        mne_epochs = mne.EpochsArray(data, info)
        
        # Create realistic metadata
        metadata = pd.DataFrame({
            'animal_id': np.repeat([1, 2, 3, 4, 5], 20),
            'genotype': np.tile(['WT', 'KO'], 50),
            'condition': np.tile(['baseline', 'treatment'], 50),
            'session': np.random.choice(['session1', 'session2'], n_epochs),
            'feature1': np.random.randn(n_epochs),
            'feature2': np.random.randn(n_epochs),
        })
        mne_epochs.metadata = metadata
        
        # Create Epochs object
        epochs = Epochs(mne_epochs, non_feature_cols=['animal_id', 'genotype', 'condition', 'session'])
        epochs.get_features()  # Initialize features
        
        return epochs
    
    def test_basic_functionality(self, sample_epochs):
        """Test basic PSD comparison functionality."""
        fig, axes = sample_epochs.compare_psd(hue='genotype', plot_type='line')
        
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 4  # Should have 4 channel subplots
        assert all(hasattr(ax, 'plot') for ax in axes.ravel())
        plt.close(fig)
    
    def test_single_hue_parameter(self, sample_epochs):
        """Test with single hue parameter."""
        fig, axes = sample_epochs.compare_psd(hue='genotype', channels=['C3', 'C4'])
        
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 2  # Should have 2 channel subplots
        plt.close(fig)
    
    def test_multiple_hue_parameters(self, sample_epochs):
        """Test with multiple hue parameters."""
        fig, axes = sample_epochs.compare_psd(hue=['genotype', 'condition'])
        
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 4  # Should have 4 channel subplots
        plt.close(fig)
    
    def test_different_channels_parameter(self, sample_epochs):
        """Test different channel specifications."""
        # Test with 'all' channels
        fig1, axes1 = sample_epochs.compare_psd(hue='genotype', channels='all')
        assert len(axes1) == 4
        plt.close(fig1)
        
        # Test with specific channel list
        fig2, axes2 = sample_epochs.compare_psd(hue='genotype', channels=['C3', 'Cz'])
        assert len(axes2) == 2
        plt.close(fig2)
    
    def test_different_methods(self, sample_epochs):
        """Test different PSD computation methods."""
        # Test multitaper method
        fig1, axes1 = sample_epochs.compare_psd(hue='genotype', method='multitaper')
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test Welch method
        fig2, axes2 = sample_epochs.compare_psd(hue='genotype', method='welch')
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
    
    def test_avg_level_parameter(self, sample_epochs):
        """Test different averaging levels."""
        # Test subject-level averaging
        fig1, axes1 = sample_epochs.compare_psd(hue='genotype', avg_level='subject')
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test all-epochs averaging
        fig2, axes2 = sample_epochs.compare_psd(hue='genotype', avg_level='all')
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
    
    def test_plot_types(self, sample_epochs):
        """Test different plot types."""
        plot_types = ['line', 'spectrum', 'box', 'bar', 'violin']
        
        for plot_type in plot_types:
            fig, axes = sample_epochs.compare_psd(hue='genotype', plot_type=plot_type)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_normalize_parameter(self, sample_epochs):
        """Test normalization parameter."""
        # Test with normalization
        fig1, axes1 = sample_epochs.compare_psd(hue='genotype', normalize=True)
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test without normalization
        fig2, axes2 = sample_epochs.compare_psd(hue='genotype', normalize=False)
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
    
    def test_err_method_parameter(self, sample_epochs):
        """Test different error methods."""
        err_methods = ['sd', 'sem', 'ci', None]
        
        for err_method in err_methods:
            fig, axes = sample_epochs.compare_psd(hue='genotype', err_method=err_method)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_plot_individuals_parameter(self, sample_epochs):
        """Test plot individuals parameter."""
        fig, axes = sample_epochs.compare_psd(
            hue='genotype', 
            plot_individuals=True,
            avg_level='subject'
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_hue_order_parameter(self, sample_epochs):
        """Test explicit hue ordering."""
        # Single hue with order
        fig1, axes1 = sample_epochs.compare_psd(
            hue='genotype', 
            hue_order=['KO', 'WT']
        )
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Multiple hue with order
        fig2, axes2 = sample_epochs.compare_psd(
            hue=['genotype', 'condition'],
            hue_order=[('WT', 'baseline'), ('KO', 'treatment')]
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
    
    def test_freq_bands_parameter(self, sample_epochs):
        """Test custom frequency bands."""
        custom_bands = {
            'low': (1, 8),
            'high': (8, 30)
        }
        
        fig, axes = sample_epochs.compare_psd(
            hue='genotype',
            plot_type='box',
            freq_bands=custom_bands
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_bad_channels_parameter(self, sample_epochs):
        """Test bad channels handling."""
        bad_channels = {
            1: ['C3'],  # Animal 1 has bad C3
            None: ['Fz']  # Global bad channel
        }
        
        fig, axes = sample_epochs.compare_psd(
            hue='genotype',
            bad_channels=bad_channels
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_stats_parameter(self, sample_epochs):
        """Test statistical analysis."""
        # Line plot with stats (requires exactly 2 groups)
        fig, axes, stats_df = sample_epochs.compare_psd(
            hue='genotype',
            plot_type='line',
            stats=True
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(stats_df, pd.DataFrame)
        assert 'p_value' in stats_df.columns
        plt.close(fig)
        
        # Categorical plot with stats
        fig, axes, stats_df = sample_epochs.compare_psd(
            hue='genotype',
            plot_type='box',
            stats=True
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(stats_df, pd.DataFrame)
        plt.close(fig)
    
    def test_return_df_parameter(self, sample_epochs):
        """Test returning underlying DataFrame."""
        # Line plot with return_df
        result = sample_epochs.compare_psd(
            hue='genotype',
            plot_type='line',
            return_df=True
        )
        fig, axes, data_df = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(data_df, pd.DataFrame)
        assert 'freq' in data_df.columns
        plt.close(fig)
        
        # Categorical plot with return_df
        result = sample_epochs.compare_psd(
            hue='genotype',
            plot_type='box',
            return_df=True
        )
        fig, axes, data_df = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(data_df, pd.DataFrame)
        assert 'band' in data_df.columns
        plt.close(fig)
    
    def test_kwargs_passing(self, sample_epochs):
        """Test that kwargs are passed to PSD computation."""
        # Test with Welch-specific parameters
        fig, axes = sample_epochs.compare_psd(
            hue='genotype',
            method='welch',
            n_fft=512,
            n_per_seg=256
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_caching_behavior(self, sample_epochs):
        """Test that results are cached properly."""
        # First call
        fig1, axes1 = sample_epochs.compare_psd(hue='genotype')
        plt.close(fig1)
        
        # Check cache exists
        assert hasattr(sample_epochs, 'psd_results')
        assert len(sample_epochs.psd_results) > 0
        
        # Second call with same parameters should use cache
        fig2, axes2 = sample_epochs.compare_psd(hue='genotype')
        plt.close(fig2)
        
        # Cache should not have grown
        cache_size_after = len(sample_epochs.psd_results)
        assert cache_size_after == 1  # Should still be 1 entry
    
    def test_error_conditions(self, sample_epochs):
        """Test various error conditions."""
        # Invalid hue column
        with pytest.raises(ValueError, match="Hue column.*not in metadata"):
            sample_epochs.compare_psd(hue='invalid_column')
        
        # Invalid avg_level
        with pytest.raises(ValueError, match="avg_level must be"):
            sample_epochs.compare_psd(hue='genotype', avg_level='invalid')
        
        # Invalid plot_type
        with pytest.raises(ValueError, match="plot_type must be"):
            sample_epochs.compare_psd(hue='genotype', plot_type='invalid')
        
        # Stats with more than 2 groups - create explicit 3-group scenario
        n_epochs = len(sample_epochs.metadata)
        # Ensure we have exactly 3 groups with reasonable distribution
        group_size = n_epochs // 3
        three_groups = (['GroupA'] * group_size + 
                    ['GroupB'] * group_size + 
                    ['GroupC'] * (n_epochs - 2 * group_size))
        
        sample_epochs.metadata['three_groups'] = three_groups
        sample_epochs.epochs.metadata = sample_epochs.metadata
        
        # Verify we actually have 3 groups
        unique_groups = sample_epochs.metadata['three_groups'].unique()
        assert len(unique_groups) == 3, f"Expected 3 groups, got {len(unique_groups)}"
        
        # Now test that stats fails with >2 groups
        with pytest.raises(ValueError, match="requires exactly 2 groups"):
            sample_epochs.compare_psd(hue='three_groups', stats=True)
    
    def test_missing_animal_id_error(self, sample_epochs):
        """Test error when animal_id is missing for subject-level analysis."""
        # Remove animal_id column
        sample_epochs.metadata = sample_epochs.metadata.drop(columns=['animal_id'])
        sample_epochs.epochs.metadata = sample_epochs.metadata
        
        with pytest.raises(ValueError, match="requires 'animal_id'"):
            sample_epochs.compare_psd(hue='genotype', avg_level='subject')


class TestCompareCon:
    """Comprehensive unit tests for the compare_con method."""
    
    @pytest.fixture
    def sample_epochs(self):
        """Create a sample Epochs object for testing."""
        # Create mock MNE epochs with realistic metadata
        info = mne.create_info(['C3', 'C4', 'Cz', 'Fz'], 250, 'eeg')
        n_epochs = 50  # Smaller for connectivity tests
        n_channels = 4
        n_times = 250  # 1 second at 250 Hz
        
        # Create synthetic EEG data with some connectivity
        np.random.seed(42)
        data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
        
        # Add correlated signals between C3 and C4
        times = np.arange(n_times) / 250
        common_signal = np.sin(2 * np.pi * 10 * times)  # 10 Hz
        
        for epoch in range(n_epochs):
            data[epoch, 0, :] += 2e-6 * common_signal  # C3
            data[epoch, 1, :] += 1.5e-6 * common_signal  # C4 (correlated)
        
        mne_epochs = mne.EpochsArray(data, info)
        
        # Create realistic metadata
        metadata = pd.DataFrame({
            'animal_id': np.repeat([1, 2, 3, 4, 5], 10),
            'genotype': np.tile(['WT', 'KO'], 25),
            'condition': np.tile(['baseline', 'treatment'], 25),
            'batch': np.random.choice(['batch1', 'batch2'], n_epochs),
            'feature1': np.random.randn(n_epochs),
            'feature2': np.random.randn(n_epochs),
        })
        mne_epochs.metadata = metadata
        
        # Create Epochs object
        epochs = Epochs(mne_epochs, non_feature_cols=['animal_id', 'genotype', 'condition', 'batch'])
        epochs.get_features()  # Initialize features
        
        return epochs
    
    def test_basic_bivariate_functionality(self, sample_epochs):
        """Test basic bivariate connectivity functionality."""
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap'
        )
        
        assert isinstance(fig, plt.Figure)
        assert hasattr(axes, '__iter__')  # Should be array of axes
        plt.close(fig)
    
    def test_basic_multivariate_functionality(self, sample_epochs):
        """Test basic multivariate connectivity functionality."""
        fig, ax = sample_epochs.compare_con(
            hue='genotype',
            method='mim',
            plot_type='spectrum'
        )
        
        assert isinstance(fig, plt.Figure)
        assert hasattr(ax, 'plot')  # Should be single axis
        plt.close(fig)
    
    def test_bivariate_methods(self, sample_epochs):
        """Test different bivariate connectivity methods."""
        bivariate_methods = ['coh', 'plv', 'wpli2_debiased', 'pli']
        
        for method in bivariate_methods:
            fig, axes = sample_epochs.compare_con(
                hue='genotype',
                method=method,
                plot_type='heatmap'
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_multivariate_methods(self, sample_epochs):
        """Test different multivariate connectivity methods."""
        multivariate_methods = ['mim', 'mic']  # Skip gc methods due to complexity
        
        for method in multivariate_methods:
            fig, ax = sample_epochs.compare_con(
                hue='genotype',
                method=method,
                plot_type='spectrum'
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_bivariate_plot_types(self, sample_epochs):
        """Test different bivariate plot types."""
        plot_types = ['heatmap', 'box', 'bar', 'violin', 'spectrum']
        
        for plot_type in plot_types:
            fig, axes = sample_epochs.compare_con(
                hue='genotype',
                method='wpli2_debiased',
                plot_type=plot_type
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_multivariate_plot_types(self, sample_epochs):
        """Test different multivariate plot types."""
        plot_types = ['spectrum', 'box', 'bar', 'violin']
        
        for plot_type in plot_types:
            fig, ax = sample_epochs.compare_con(
                hue='genotype',
                method='mim',
                plot_type=plot_type
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_avg_level_parameter(self, sample_epochs):
        """Test different averaging levels."""
        # Test all-epochs averaging
        fig1, axes1 = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap',
            avg_level='all'
        )
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Test subject-level averaging
        fig2, axes2 = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap',
            avg_level='subject'
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
    
    def test_hue_order_parameter(self, sample_epochs):
        """Test explicit hue ordering."""
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap',
            hue_order=['KO', 'WT']
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_node_order_parameter(self, sample_epochs):
        """Test explicit node ordering."""
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap',
            node_order=['Fz', 'Cz', 'C4', 'C3']
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_freq_bands_parameter(self, sample_epochs):
        """Test custom frequency bands."""
        custom_bands = {
            'low': (2, 8),
            'high': (8, 30)
        }
        
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='box',
            freq_bands=custom_bands
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_bad_channels_bivariate(self, sample_epochs):
        """Test bad channels handling for bivariate methods."""
        bad_channels = {
            1: ['C3'],  # Animal 1 has bad C3
            None: ['Fz']  # Global bad channel
        }
        
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap',
            bad_channels=bad_channels,
            avg_level='subject'
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_bad_channels_multivariate(self, sample_epochs):
        """Test bad channels handling for multivariate methods."""
        bad_channels = {
            1: ['C3'],  # Animal 1 has bad C3
        }
        
        fig, ax = sample_epochs.compare_con(
            hue='genotype',
            method='mim',
            plot_type='spectrum',
            bad_channels=bad_channels,
            avg_level='subject'
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_multivariate_nodes_parameter(self, sample_epochs):
        """Test multivariate nodes selection."""
        fig, ax = sample_epochs.compare_con(
            hue='genotype',
            method='mim',
            plot_type='spectrum',
            multivariate_nodes=['C3', 'C4', 'Cz']
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_individuals_parameter(self, sample_epochs):
        """Test plot individuals parameter."""
        # Bivariate
        fig1, axes1 = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='box',
            plot_individuals=True,
            avg_level='subject'
        )
        assert isinstance(fig1, plt.Figure)
        plt.close(fig1)
        
        # Multivariate
        fig2, ax2 = sample_epochs.compare_con(
            hue='genotype',
            method='mim',
            plot_type='box',
            plot_individuals=True,
            avg_level='subject'
        )
        assert isinstance(fig2, plt.Figure)
        plt.close(fig2)
    
    def test_stats_parameter_bivariate(self, sample_epochs):
        """Test statistical analysis for bivariate methods."""
        fig, axes, stats_df = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='box',
            stats='ttest',
            avg_level='subject'
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(stats_df, pd.DataFrame)
        assert 'p_value' in stats_df.columns
        plt.close(fig)
    
    def test_stats_parameter_multivariate(self, sample_epochs):
        """Test statistical analysis for multivariate methods."""
        fig, ax, stats_df = sample_epochs.compare_con(
            hue='genotype',
            method='mim',
            plot_type='box',
            stats='auto',
            avg_level='subject'
        )
        assert isinstance(fig, plt.Figure)
        assert isinstance(stats_df, pd.DataFrame)
        assert 'p_value' in stats_df.columns
        plt.close(fig)
    
    def test_diff_plot_type(self, sample_epochs):
        """Test difference plot type."""
        # Add batch column for mixed effects (required for diff plot)
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='diff',
            stats='ttest'
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_err_method_parameter(self, sample_epochs):
        """Test different error methods for spectrum plots."""
        err_methods = ['sd', 'sem', 'ci', None]
        
        for err_method in err_methods:
            fig, ax = sample_epochs.compare_con(
                hue='genotype',
                method='mim',
                plot_type='spectrum',
                err_method=err_method,
                avg_level='subject'
            )
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
    
    def test_heatmap_parameters(self, sample_epochs):
        """Test heatmap-specific parameters."""
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap',
            vmin=0.1,
            vmax=0.9,
            upper=True
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_categorical_parameters(self, sample_epochs):
        """Test categorical plot-specific parameters."""
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='box',
            ylims=(0, 1)
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_con_kwargs_parameter(self, sample_epochs):
        """Test passing additional connectivity parameters."""
        fig, axes = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap',
            con_kwargs={'n_jobs': 1}
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_caching_behavior(self, sample_epochs):
        """Test that connectivity results are cached properly."""
        # First call
        fig1, axes1 = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap'
        )
        plt.close(fig1)
        
        # Check cache exists
        assert hasattr(sample_epochs, 'con_results')
        assert len(sample_epochs.con_results) > 0
        
        # Second call with same parameters should use cache
        fig2, axes2 = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap'
        )
        plt.close(fig2)
    
    def test_error_conditions(self, sample_epochs):
        """Test various error conditions."""
        # Invalid hue column
        with pytest.raises(ValueError, match="Hue column.*not in metadata"):
            sample_epochs.compare_con(
                hue='invalid_column',
                method='wpli2_debiased',
                plot_type='heatmap'
            )
        
        # Invalid avg_level
        with pytest.raises(ValueError, match="avg_level must be"):
            sample_epochs.compare_con(
                hue='genotype',
                method='wpli2_debiased',
                plot_type='heatmap',
                avg_level='invalid'
            )
        
        # Invalid plot_type for bivariate
        with pytest.raises(ValueError, match="not valid for bivariate"):
            sample_epochs.compare_con(
                hue='genotype',
                method='wpli2_debiased',
                plot_type='invalid_type'
            )
        
        # Invalid plot_type for multivariate
        with pytest.raises(ValueError, match="not valid for multivariate"):
            sample_epochs.compare_con(
                hue='genotype',
                method='mim',
                plot_type='heatmap'  # Not allowed for multivariate
            )
        
        # Create proper 3-group scenario
        n_epochs = len(sample_epochs.metadata)
        sample_epochs.metadata['three_groups'] = (['A'] * (n_epochs//3) + 
                                                ['B'] * (n_epochs//3) + 
                                                ['C'] * (n_epochs - 2*(n_epochs//3)))
        sample_epochs.epochs.metadata = sample_epochs.metadata
        
        with pytest.raises(ValueError, match="requires exactly 2 groups"):
            sample_epochs.compare_psd(hue='three_groups', stats=True)
        
    def test_missing_animal_id_error(self, sample_epochs):
        """Test error when animal_id is missing for subject-level analysis."""
        # Remove animal_id column
        sample_epochs.metadata = sample_epochs.metadata.drop(columns=['animal_id'])
        sample_epochs.epochs.metadata = sample_epochs.metadata
        
        with pytest.raises(ValueError, match="requires 'animal_id'"):
            sample_epochs.compare_con(
                hue='genotype',
                method='wpli2_debiased',
                plot_type='heatmap',
                avg_level='subject'
            )
    
    def test_no_channels_after_bad_removal(self, sample_epochs):
        """Test error when no channels remain after bad channel removal."""
        bad_channels = {
            None: ['C3', 'C4', 'Cz', 'Fz']  # Remove all channels
        }
        
        with pytest.raises(ValueError, match="no channels remain"):
            sample_epochs.compare_con(
                hue='genotype',
                method='mim',
                plot_type='spectrum',
                bad_channels=bad_channels
            )


class TestIntegration:
    """Integration tests for both methods together."""
    
    @pytest.fixture
    def sample_epochs(self):
        """Create a comprehensive sample Epochs object."""
        info = mne.create_info(['C3', 'C4', 'Cz', 'Fz', 'O1', 'O2'], 250, 'eeg')
        n_epochs = 80
        n_channels = 6
        n_times = 500
        
        np.random.seed(42)
        data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
        mne_epochs = mne.EpochsArray(data, info)
        
        metadata = pd.DataFrame({
            'animal_id': np.repeat([1, 2, 3, 4], 20),
            'genotype': np.tile(['WT', 'KO'], 40),
            'condition': np.tile(['baseline', 'treatment'], 40),
            'session': np.tile(['session1', 'session2'], 40),
            'feature1': np.random.randn(n_epochs),
            'feature2': np.random.randn(n_epochs),
        })
        mne_epochs.metadata = metadata
        
        epochs = Epochs(mne_epochs, non_feature_cols=['animal_id', 'genotype', 'condition', 'session'])
        epochs.get_features()
        
        return epochs
    
    def test_both_methods_use_same_bad_channels(self, sample_epochs):
        """Test that both methods handle bad_channels consistently."""
        bad_channels = {1: ['C3'], None: ['O2']}
        
        # PSD analysis
        fig1, axes1 = sample_epochs.compare_psd(
            hue='genotype',
            bad_channels=bad_channels
        )
        plt.close(fig1)
        
        # Connectivity analysis
        fig2, axes2 = sample_epochs.compare_con(
            hue='genotype',
            method='wpli2_debiased',
            plot_type='heatmap',
            bad_channels=bad_channels
        )
        plt.close(fig2)
        
        # Both should complete without errors
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)
    
    def test_both_methods_consistent_hue_handling(self, sample_epochs):
        """Test that both methods handle hue parameters consistently."""
        # Single hue
        fig1, axes1 = sample_epochs.compare_psd(hue='genotype')
        fig2, axes2 = sample_epochs.compare_con(
            hue='genotype', method='wpli2_debiased', plot_type='heatmap'
        )
        plt.close(fig1)
        plt.close(fig2)
        
        # Multiple hue
        fig3, axes3 = sample_epochs.compare_psd(hue=['genotype', 'condition'])
        fig4, axes4 = sample_epochs.compare_con(
            hue=['genotype', 'condition'], method='wpli2_debiased', plot_type='heatmap'
        )
        plt.close(fig3)
        plt.close(fig4)
        
        # All should complete successfully
        for fig in [fig1, fig2, fig3, fig4]:
            assert isinstance(fig, plt.Figure)
    
    def test_both_methods_caching_independence(self, sample_epochs):
        """Test that caching systems don't interfere with each other."""
        # Run PSD analysis
        fig1, axes1 = sample_epochs.compare_psd(hue='genotype')
        plt.close(fig1)
        
        # Run connectivity analysis
        fig2, axes2 = sample_epochs.compare_con(
            hue='genotype', method='wpli2_debiased', plot_type='heatmap'
        )
        plt.close(fig2)
        
        # Check both have their own caches
        assert hasattr(sample_epochs, 'psd_results')
        assert hasattr(sample_epochs, 'con_results')
        assert len(sample_epochs.psd_results) > 0
        assert len(sample_epochs.con_results) > 0
    
    def test_parameter_validation_consistency(self, sample_epochs):
        """Test that parameter validation is consistent between methods."""
        # Both should reject invalid hue columns
        with pytest.raises(ValueError, match="Hue column.*not in metadata"):
            sample_epochs.compare_psd(hue='invalid')
        
        with pytest.raises(ValueError, match="Hue column.*not in metadata"):
            sample_epochs.compare_con(
                hue='invalid', method='wpli2_debiased', plot_type='heatmap'
            )
        
        # Both should reject invalid avg_level
        with pytest.raises(ValueError, match="avg_level must be"):
            sample_epochs.compare_psd(hue='genotype', avg_level='invalid')
        
        with pytest.raises(ValueError, match="avg_level must be"):
            sample_epochs.compare_con(
                hue='genotype', method='wpli2_debiased', 
                plot_type='heatmap', avg_level='invalid'
            )


# Additional test utilities and fixtures
@pytest.fixture
def mock_epochs_no_features():
    """Create epochs object without features for testing error conditions."""
    info = mne.create_info(['C3', 'C4'], 250, 'eeg')
    data = np.random.randn(10, 2, 250) * 1e-6
    mne_epochs = mne.EpochsArray(data, info)
    
    metadata = pd.DataFrame({
        'genotype': ['WT'] * 5 + ['KO'] * 5,
        'condition': ['baseline', 'treatment'] * 5
    })
    mne_epochs.metadata = metadata
    
    epochs = Epochs(mne_epochs)
    # Don't call get_features() to test error condition
    return epochs


class TestErrorHandling:
    """Test comprehensive error handling for both methods."""
    
    def test_no_features_error_psd(self, mock_epochs_no_features):
        """Test error when features haven't been computed for PSD."""
        # This should work because compare_psd doesn't require pre-computed features
        fig, axes = mock_epochs_no_features.compare_psd(hue='genotype')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_no_features_error_con(self, mock_epochs_no_features):
        """Test that connectivity works without pre-computed features."""
        # This should work because compare_con doesn't require pre-computed features
        fig, axes = mock_epochs_no_features.compare_con(
            hue='genotype', method='wpli2_debiased', plot_type='heatmap'
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_empty_groups_error(self):
        """Test handling of empty groups after filtering."""
        # Create epochs with no matching data for some conditions
        info = mne.create_info(['C3', 'C4'], 250, 'eeg')
        data = np.random.randn(5, 2, 250) * 1e-6
        mne_epochs = mne.EpochsArray(data, info)
        
        metadata = pd.DataFrame({
            'genotype': ['WT'] * 5,  # Only WT, no KO
            'animal_id': [1, 1, 2, 2, 3]
        })
        mne_epochs.metadata = metadata
        
        epochs = Epochs(mne_epochs)
        epochs.get_features()
        
        # This should still work with single group
        fig, axes = epochs.compare_psd(hue='genotype')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPerformance:
    """Test performance-related aspects."""
    
    def test_caching_performance_psd(self, sample_epochs):
        """Test that caching improves performance for PSD."""
        import time
        
        # First call (no cache)
        start_time = time.time()
        fig1, axes1 = sample_epochs.compare_psd(hue='genotype')
        first_call_time = time.time() - start_time
        plt.close(fig1)
        
        # Second call (with cache)
        start_time = time.time()
        fig2, axes2 = sample_epochs.compare_psd(hue='genotype')
        second_call_time = time.time() - start_time
        plt.close(fig2)
        
        # Second call should be faster (or at least not much slower)
        assert second_call_time <= first_call_time * 1.5  # Allow some variance
    
    def test_caching_performance_con(self, sample_epochs):
        """Test that caching improves performance for connectivity."""
        import time
        
        # First call (no cache)
        start_time = time.time()
        fig1, axes1 = sample_epochs.compare_con(
            hue='genotype', method='wpli2_debiased', plot_type='heatmap'
        )
        first_call_time = time.time() - start_time
        plt.close(fig1)
        
        # Second call (with cache)
        start_time = time.time()
        fig2, axes2 = sample_epochs.compare_con(
            hue='genotype', method='wpli2_debiased', plot_type='heatmap'
        )
        second_call_time = time.time() - start_time
        plt.close(fig2)
        
        # Second call should be faster
        assert second_call_time <= first_call_time * 1.5


class TestDataTypesAndFormats:
    """Test different data types and formats."""
    
    def test_different_metadata_dtypes(self):
        """Test handling of different metadata data types."""
        info = mne.create_info(['C3', 'C4'], 250, 'eeg')
        data = np.random.randn(20, 2, 250) * 1e-6
        mne_epochs = mne.EpochsArray(data, info)
        
        # Mix of data types in metadata
        metadata = pd.DataFrame({
            'animal_id': np.array([1, 2, 3, 4] * 5, dtype=int),
            'genotype': pd.Categorical(['WT', 'KO'] * 10),
            'condition': np.array(['baseline', 'treatment'] * 10),
            'score': np.random.rand(20).astype(np.float32),
            'trial_num': np.arange(20, dtype=np.int64)
        })
        mne_epochs.metadata = metadata
        
        epochs = Epochs(mne_epochs, non_feature_cols=['animal_id', 'genotype', 'condition'])
        epochs.get_features(['score', 'trial_num'])
        
        # Should handle different dtypes gracefully
        fig, axes = epochs.compare_psd(hue='genotype')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_missing_values_handling(self):
        """Test handling of missing values in metadata."""
        info = mne.create_info(['C3', 'C4'], 250, 'eeg')
        data = np.random.randn(20, 2, 250) * 1e-6
        mne_epochs = mne.EpochsArray(data, info)
        
        # Include some NaN values
        genotype = ['WT', 'KO'] * 10
        genotype[5] = np.nan  # Missing value
        
        metadata = pd.DataFrame({
            'animal_id': [1, 2, 3, 4] * 5,
            'genotype': genotype,
            'condition': ['baseline', 'treatment'] * 10,
            'feature1': np.random.randn(20)
        })
        mne_epochs.metadata = metadata
        
        epochs = Epochs(mne_epochs, non_feature_cols=['animal_id', 'genotype', 'condition'])
        epochs.get_features(['feature1'])
        
        # Should handle NaN values appropriately
        # Note: This might raise an error or handle gracefully depending on implementation
        try:
            fig, axes = epochs.compare_psd(hue='genotype')
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except (ValueError, KeyError):
            # Expected behavior for NaN in grouping variables
            pass


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_epoch(self):
        """Test with single epoch."""
        info = mne.create_info(['C3', 'C4'], 250, 'eeg')
        data = np.random.randn(1, 2, 250) * 1e-6  # Single epoch
        mne_epochs = mne.EpochsArray(data, info)
        
        metadata = pd.DataFrame({
            'genotype': ['WT'],
            'animal_id': [1],
            'feature1': [0.5]
        })
        mne_epochs.metadata = metadata
        
        epochs = Epochs(mne_epochs, non_feature_cols=['genotype', 'animal_id'])
        epochs.get_features(['feature1'])
        
        # Should handle single epoch gracefully
        fig, axes = epochs.compare_psd(hue='genotype')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_single_channel(self):
        """Test with single channel."""
        info = mne.create_info(['C3'], 250, 'eeg')
        data = np.random.randn(20, 1, 250) * 1e-6
        mne_epochs = mne.EpochsArray(data, info)
        
        metadata = pd.DataFrame({
            'genotype': ['WT', 'KO'] * 10,
            'animal_id': [1, 2] * 10,
            'feature1': np.random.randn(20)
        })
        mne_epochs.metadata = metadata
        
        epochs = Epochs(mne_epochs, non_feature_cols=['genotype', 'animal_id'])
        epochs.get_features(['feature1'])
        
        # PSD should work with single channel
        fig, axes = epochs.compare_psd(hue='genotype')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Connectivity should fail or handle gracefully with single channel
        try:
            fig, axes = epochs.compare_con(
                hue='genotype', method='wpli2_debiased', plot_type='heatmap'
            )
            plt.close(fig)
        except (ValueError, RuntimeError):
            # Expected for single channel connectivity
            pass
    
    def test_very_short_epochs(self):
        """Test with very short epochs."""
        info = mne.create_info(['C3', 'C4'], 250, 'eeg')
        data = np.random.randn(20, 2, 50) * 1e-6  # Very short epochs (0.2 seconds)
        mne_epochs = mne.EpochsArray(data, info)
        
        metadata = pd.DataFrame({
            'genotype': ['WT', 'KO'] * 10,
            'animal_id': [1, 2, 3, 4] * 5,
            'feature1': np.random.randn(20)
        })
        mne_epochs.metadata = metadata
        
        epochs = Epochs(mne_epochs, non_feature_cols=['genotype', 'animal_id'])
        epochs.get_features(['feature1'])
        
        # Should handle short epochs, possibly with warnings
        try:
            fig, axes = epochs.compare_psd(hue='genotype', method='welch')
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
        except ValueError:
            # Expected for very short epochs with Welch method
            pass
    
    def test_identical_groups(self):
        """Test with identical data in different groups."""
        info = mne.create_info(['C3', 'C4'], 250, 'eeg')
        # Same data for all epochs
        base_data = np.random.randn(1, 2, 250) * 1e-6
        data = np.tile(base_data, (20, 1, 1))
        mne_epochs = mne.EpochsArray(data, info)
        
        metadata = pd.DataFrame({
            'genotype': ['WT', 'KO'] * 10,
            'animal_id': [1, 2, 3, 4] * 5,
            'feature1': [1.0] * 20  # Identical features
        })
        mne_epochs.metadata = metadata
        
        epochs = Epochs(mne_epochs, non_feature_cols=['genotype', 'animal_id'])
        epochs.get_features(['feature1'])
        
        # Should handle identical data gracefully
        fig, axes = epochs.compare_psd(hue='genotype')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# Test configuration and setup
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Helper functions for test data generation
def create_realistic_eeg_data(n_epochs, n_channels, n_times, sfreq=250):
    """Create realistic EEG data with known spectral characteristics."""
    np.random.seed(42)
    times = np.arange(n_times) / sfreq
    data = np.zeros((n_epochs, n_channels, n_times))
    
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            # Base noise
            data[epoch, ch, :] = np.random.randn(n_times) * 1e-6
            
            # Add physiological rhythms
            # Alpha (8-12 Hz)
            alpha_freq = 10 + np.random.randn() * 1
            data[epoch, ch, :] += 2e-6 * np.sin(2 * np.pi * alpha_freq * times)
            
            # Theta (4-8 Hz)
            theta_freq = 6 + np.random.randn() * 0.5
            data[epoch, ch, :] += 1e-6 * np.sin(2 * np.pi * theta_freq * times)
            
            # Beta (13-30 Hz)
            beta_freq = 20 + np.random.randn() * 3
            data[epoch, ch, :] += 0.5e-6 * np.sin(2 * np.pi * beta_freq * times)
    
    return data


def create_connected_eeg_data(n_epochs, n_channels, n_times, sfreq=250):
    """Create EEG data with known connectivity patterns."""
    np.random.seed(42)
    times = np.arange(n_times) / sfreq
    data = np.zeros((n_epochs, n_channels, n_times))
    
    # Create common source signal
    source_freq = 10  # 10 Hz
    common_signal = np.sin(2 * np.pi * source_freq * times)
    
    for epoch in range(n_epochs):
        # Add noise to each channel
        for ch in range(n_channels):
            data[epoch, ch, :] = np.random.randn(n_times) * 1e-6
        
        # Add connected components
        # Channels 0 and 1 are strongly connected
        data[epoch, 0, :] += 2e-6 * common_signal
        data[epoch, 1, :] += 1.8e-6 * common_signal + 0.1e-6 * np.random.randn(n_times)
        
        # Channels 2 and 3 have weaker connection
        if n_channels > 3:
            delayed_signal = np.roll(common_signal, 10)  # Small delay
            data[epoch, 2, :] += 1e-6 * delayed_signal
            data[epoch, 3, :] += 0.8e-6 * delayed_signal + 0.2e-6 * np.random.randn(n_times)
    
    return data


# Performance benchmarking utilities
class BenchmarkTimer:
    """Simple timer for benchmarking test performance."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


# Example usage and test discovery
if __name__ == "__main__":
    # Run specific test classes
    pytest.main([
        __file__ + "::TestComparePSD",
        __file__ + "::TestCompareCon", 
        __file__ + "::TestIntegration",
        "-v", "--tb=short"
    ])