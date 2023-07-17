import matplotlib.pyplot as plt
import matplotlib.cbook
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns
from .eeg_utils import add_virtual_timestamps
import warnings 
from pytz import timezone as pytz_timezone
from datetime import datetime

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

def fill_area_in_plot(list_areas:dict, plt_obj:plt.Axes, label:str ='Stimulation', t_ax:np.ndarray=None) -> None:
    """
    Fill multiple areas of a plot object with a single label  

    Parameters
    ----------
    list_areas  : dict
                    Dict of tuples which contain the beginning and end (datetime objects) of the area to be filled
    plt_obj     : matplotlib.axes.Axes or matplotlib.plt
                    The plot object where the areas will be colored
    label       : str
                    label of the areas
    t_ax        : numpy.ndarray
                    Time array specifying the range for the plot

    """
    if t_ax is not None:
        min_t_ax, max_t_ax = np.min(t_ax), np.max(t_ax)

    else:
        min_t_ax, max_t_ax = mdates.num2date(plt_obj.get_xlim())

    for stimulator_name, stimulator_limits in list_areas.items():
        if stimulator_limits:
            for ind, (lower_lim, upper_lim) in enumerate(stimulator_limits):
                if (lower_lim < min_t_ax and upper_lim < min_t_ax) or (lower_lim > max_t_ax and upper_lim > max_t_ax):
                    continue  # skip if the span is out of the t_ax range
                
                # Adjust the span to the t_ax range if necessary
                lower_lim = max(lower_lim, min_t_ax)
                upper_lim = min(upper_lim, max_t_ax)

                label = stimulator_name if ind==0 else ''  # Just label the first area
                
                plt_obj.axvspan(lower_lim, upper_lim, color='C07', label=label, alpha=0.2, hatch='.')
                
    return

def format_axes(ax,grid_lines='major',grid_color='gray',grid_style='--'):
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which=grid_lines,linestyle=grid_style, linewidth=0.5, color=grid_color, alpha=0.5)

    # Remove the left and bottom spines by setting their color to 'none'
    # ax.spines['left'].set_color('none')
    # ax.spines['bottom'].set_color('none')

def plot_and_fill(ax, t_ax, data_list, plot_nb, total_plots, ylabel, format_axes_pars, line_opts_list, fill_areas):
    
    format_axes(ax,**format_axes_pars)
    ax.set_ylabel(ylabel, fontsize=15)
    if int(str(plot_nb)[-1]) == total_plots:
        ax.set_xlabel('Time', fontsize=15)

    for data, line_opts in zip(data_list, line_opts_list):
        ax.plot(t_ax, data, **line_opts)

    # Get the timezone, if no data use UTC
    if len(t_ax):
        tz = t_ax.dt.tz
    else:
        timezone = 'UTC'
        tz = pytz_timezone(timezone)

    # Use DateFormatter to format the x-axis labels, using the specified timezone
    date_format = DateFormatter('%H:%M:%S', tz=tz)
    ax.xaxis.set_major_formatter(date_format)

    if fill_areas:
        fill_area_in_plot(fill_areas, ax,t_ax)

def plot_data(fig,
             data_dict:dict,
             sampling_f:dict,
             window_length={'ecg':0,'hr':0,'acc':0,'br':0,'bmag':0},
             stimulation_areas={},
             timezone = pytz_timezone('UTC')):
    
    # Only plot signals which are present in data_dict AND window_length
    total_plots = len(set(data_dict.keys()) & set(window_length.keys()))


    # Update the stimulation_areas
    stimulation_areas = verify_areas(stimulation_areas,timezone)

    # Iterate through the list of desired plots to show
    ind=0

    for plot_type,biosignal_object in data_dict.items():

        # Prepare the key and skip if this data type isn't in window length
        plot_type = plot_type.lower()
        if plot_type not in window_length.keys():
            continue
        
        # Get the sampling frequency for this plot type
        fs = sampling_f[plot_type]['sampling_f']
        
        # Window length
        wl = int(window_length[plot_type]*fs)
        
        # Remove the timestamp columns
        # data = biosignal_object.data.drop(columns='Timestamp')
        data = biosignal_object.data.drop(columns=[col for col in biosignal_object.data.columns if 'timestamp' in col.lower()])

        #Arrange plots in a single column
        plot_nb = int(f'{total_plots}1{ind+1}')

        # Time axis cut to the same length as the data 
        # t_ax = np.arange(0, data.shape[0]/ fs, step=1/fs)
        t_ax = biosignal_object.data['Timestamp']

        if t_ax.shape[0]>data.shape[0]:
            t_ax=t_ax[:-1]

        if (plot_type =='ecg'):
            plot_and_fill(ax=plt.subplot(plot_nb, label=plot_type), 
                            t_ax=t_ax[-wl:], 
                            data_list=[data[-wl:]], 
                            plot_nb = plot_nb, 
                            total_plots = total_plots, 
                            ylabel = 'ECG [$\mu$V]', 
                            format_axes_pars={'grid_lines':'both','grid_color':'r','grid_style':'-'},
                            line_opts_list = [{'color': 'k', 'label': 'ECG'}], 
                            fill_areas = stimulation_areas)


        # Plot the HR in bpm
        elif plot_type in ['hr','ecg_hr']:

            plot_and_fill(ax=plt.subplot(plot_nb, label=plot_type), 
                            t_ax=t_ax[-wl:], 
                            data_list=[data[-wl:]], 
                            plot_nb = plot_nb, 
                            total_plots = total_plots, 
                            ylabel = 'HR \n[bpm]', 
                            format_axes_pars={},
                            line_opts_list = [{'color': 'r', 'label': 'HR'}], 
                            fill_areas = stimulation_areas)

        elif plot_type =='acc': 
            # if there is data, center it 
            if len(data)>0:
                data = data - data.iloc[0]
            colors = plt.get_cmap('Pastel1')
            ax = plt.subplot(plot_nb, label='acc')
            plot_and_fill(ax=ax, 
                            t_ax = t_ax[-wl:], 
                            data_list = [data['X'][-wl:], data['Y'][-wl:], data['Z'][-wl:]], 
                            plot_nb = plot_nb, 
                            total_plots = total_plots, 
                            ylabel = 'Accel. \nAmp.', 
                            format_axes_pars = {},
                            line_opts_list = [{'color': colors(0), 'label': 'X', 'alpha': 1}, 
                                                {'color': colors(1), 'label': 'Y', 'alpha': 1}, 
                                                {'color': colors(2), 'label': 'Z', 'alpha': 1}], 
                            fill_areas = stimulation_areas)
            ax.legend(loc='best')

        # Plot the breathing rate
        elif plot_type =='br':
            
            plot_and_fill(ax=plt.subplot(plot_nb, label=plot_type), 
                            t_ax=t_ax[-wl:], 
                            data_list=[data[-wl:]], 
                            plot_nb = plot_nb, 
                            total_plots = total_plots, 
                            ylabel = 'Breathing \nRate [bpm]', 
                            format_axes_pars={},
                            line_opts_list = [{'color': 'cornflowerblue', 'label': 'Breathing Rate','linewidth':4,'linestyle':'dotted'}], 
                            fill_areas = [])

        # Plot the breathing magnitude
        elif plot_type =='bmag':

            plot_and_fill(ax=plt.subplot(plot_nb, label=plot_type), 
                            t_ax=t_ax[-wl:], 
                            data_list=[data[-wl:]], 
                            plot_nb = plot_nb, 
                            total_plots = total_plots, 
                            ylabel = 'Breathing \nAcc. Mag.', 
                            format_axes_pars={},
                            line_opts_list = [{'color': 'k', 'label': 'Breathing magnitude','linestyle':'--'}], 
                            fill_areas = [])


        elif plot_type =='eeg':
            
            # Black channels
            # data_list = [data[-wl:]]
            # line_opts_list = [{'color': 'k', 'label': 'EEG','lw':0.7,'alpha':0.5,'linestyle':'-'}]

            # Colored channels
            data_list = [data[ch_name][-wl:] for ch_name in data.columns]
            line_opts_list = [{'color': f'C{ind}', 'label': f'{ind}_{ch_name}', 'linestyle': '-', 'lw': 2, 'alpha': 0.7} 
                            for ind, ch_name in enumerate(data.columns)]   
            plot_and_fill(ax=plt.subplot(plot_nb, label=plot_type), 
                            t_ax=t_ax[-wl:], 
                            data_list=data_list, 
                            plot_nb = plot_nb, 
                            total_plots = total_plots, 
                            ylabel = 'EEG [$\mu$V]', 
                            format_axes_pars={},
                            line_opts_list = line_opts_list, 
                            fill_areas = stimulation_areas)
            plt.legend()

        elif plot_type =='eeg_filt':

            # Black channels
            # data_list = [data[-wl:]]
            # line_opts_list = [{'color': 'k', 'label': 'EEG filtered','lw':0.7,'alpha':0.5,'linestyle':'-'}]

            # Colored channels
            data_list = [data[ch_name][-wl:] for ch_name in data.columns]
            line_opts_list = [{'color': f'C{ind}', 'label': f'{ind}_{ch_name}', 'linestyle': '-', 'lw': 2, 'alpha': 0.7} 
                            for ind, ch_name in enumerate(data.columns)]   
            
            plot_and_fill(ax=plt.subplot(plot_nb, label=plot_type), 
                            t_ax=t_ax[-wl:], 
                            data_list=data_list, 
                            plot_nb = plot_nb, 
                            total_plots = total_plots, 
                            ylabel = 'EEG \nfiltered'+r' [$\mu$V]', 
                            format_axes_pars={},
                            line_opts_list = line_opts_list, 
                            fill_areas = stimulation_areas)
            plt.legend()

        elif plot_type == 'eeg_score':
            ax_ = plt.subplot(plot_nb,label='eeg_score')
            plot_eeg_score_accum(data[-wl:],ax_,ylim_window='alpha', time_offset=0)
            if plot_nb == total_plots:
                ax_.set_xlabel('Timestep',fontsize=15)

            # Fill the stimulation area
            if stimulation_areas:
                fill_area_in_plot(stimulation_areas,ax_)

        ind +=1

    return fig

def plot_eeg_score_accum(score_accum,ax:plt.Axes,ylim_window:str='alpha', time_offset:int=0):
    df = score_accum.copy()
    df = add_virtual_timestamps(df)
    cols_to_average = [column for column in df.columns if column not in ['Timestamp','Band']]
    # df['Mean'] = df[['TP9', 'AF7', 'AF8', 'TP10']].mean(axis=1)
    df['Mean'] = df[cols_to_average].mean(axis=1)
    sns.lineplot(x='Timestamp', y='Mean', hue='Band', data=df)
    
    # Set the y-limits of the plot 
    format_axes(ax,grid_lines='both',grid_color='gray',grid_style='--')
    ylim_upper = 1.1*df[df['Band']==ylim_window]['Mean'].max()
    ylim_lower = 0.9*df[df['Band']==ylim_window]['Mean'].min()
    ylim_upper = 1 if ylim_upper == 0.0 else ylim_upper

    ax.set_ylim(ylim_lower,ylim_upper)
    ax.set_ylabel(f'EEG score [%]',fontsize=15)
    formatter = FuncFormatter(to_percent)
    ax.yaxis.set_major_formatter(formatter)
    
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = "{:.1f}".format(100 * y)

    # The percent symbol needs escaping in latex
    if plt.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
    
def verify_areas(areas, timezone):
    # If the dictionary is empty, return it as is
    if not areas:
        return areas
    
    for key in areas:
        # If the value list for the key is empty, skip to the next iteration
        if not areas[key]:
            continue
        
        # Check the last sublist in the list
        last_sublist = areas[key][-1]
        
        # If the last element of the sublist is None, replace it with current time
        if last_sublist[-1] is None:
            last_sublist[-1] = datetime.now(timezone)
    return areas