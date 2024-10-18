import re
import traceback

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

def analyze_dataset(input_folder, dataset_name, files, channel_dict, tracking_channel,
                    min_len, digits, delimiter, main_output_folder, advanced_settings, tracking_interval):


    size_jump_threshold = advanced_settings["size_jump_threshold"]
    tracking_marker_jump_threshold= advanced_settings["tracking_marker_jump_threshold"]
    tracking_marker_peak_threshold=advanced_settings["tracking_marker_division_peak_threshold"]
    suffix = advanced_settings["suffix"]

    if not digits:
        delimiter = ""
    tracking_marker = channel_dict[tracking_channel]
    main_channels = [color for color in channel_dict.values() if color != tracking_marker]
    channels_to_color = channel_dict

    #input_folder = os.path.dirname(main_folder)

    subsets = [file.split(suffix)[0][-digits:] for file in files if dataset_name in file]
    #print(subsets)

    output_folder = create_folder(main_output_folder + dataset_name + "/")

    results = {"dataset": dataset_name,
               "error": None,
               "output_folder": output_folder,
               "run_complete": False}

    print(f"analysing dataset {dataset_name}")

    try:
    #print(f'output folder = {output_folder}')

        input_file_list = []
        if len(subsets)==1 and (not digits):
            input_file_list = [("", subsets[0]+suffix)]
        else:
            for subset in subsets:
                input_file_list.append((subset, f'{input_folder}/{dataset_name}{delimiter}{subset}{suffix}'))
        #print(input_file_list)
        # Standardize column labelling. Delete time series that were too short (decision in cell profiler pipeline)
        input_tables_cells = {}
        # image_tables = {}
        input_files_used = []
        for dataset, file in input_file_list:
            # print(dataset)
            # loading cell data
            if not os.path.exists(file):
                print(F"file: {file} not found")
                continue
            try:
                input_table = pd.read_csv(file, low_memory=False).iloc[3:]
                input_files_used.append((dataset, file))
                input_tables_cells[dataset] = input_table.dropna(subset=["TRACK_ID"])
                input_tables_cells[dataset].loc[:, "TRACK_ID"] = (
                    input_tables_cells[dataset]["TRACK_ID"].astype(int).astype(
                    str).apply(add_dataset_number, args=(dataset,)))

            except:
                print(f'subset {dataset} could not be loaded')
                print(f'{input_folder}')
                continue
        #print(input_tables_cells.keys())
        try:
            if len(input_tables_cells.values()) == 1:
                input_table_cleaned = list(input_tables_cells.values())[0]
            else:
                input_table_cleaned = pd.concat(input_tables_cells.values())
            # for col in input_table_cleaned.TrackObjects_Label:
            #    print(f'col in input_table_cleaned {col}')
        except:
            raise ValueError('Could not load any data set')


        # Tables with values

        signals_raw = extract_raw_time_series(input_table_cleaned, channels_to_color, min_len)

        object_sizes = extract_size_time_series(input_table_cleaned, min_len)
        #print("extracted object_sizes")
        # look at object size differences
        size_diffs = []
        for cell in object_sizes:
            for x in range(object_sizes.shape[0] - 1):
                size_diff = abs(object_sizes[cell].iloc[x + 1] - object_sizes[cell].iloc[x])
                rel_diff = size_diff / object_sizes[cell].iloc[x]
                if not np.isnan(rel_diff):
                    # print(rel_diff)
                    size_diffs.append(rel_diff)
        #print("calculated size differences")
        # From object size, calculate how the relative size of the cell has changed
        # compared to previous time point Then collect size jumps (either up = 1, or down = -1)
        object_size_diff = object_sizes.apply(difference_to_prev)
        size_jumps = object_size_diff.apply(get_jumps, args=(size_jump_threshold,), axis=0)

        #  detect iRFP signal jumps
        tracking_marker_int_diffs = signals_raw[tracking_marker].apply(difference_to_prev)
        tracking_marker_jumps = tracking_marker_int_diffs.apply(get_jumps,
                                                                args=(tracking_marker_jump_threshold,),
                                                                axis=0)

        # smooth tracking_marker signal
        def smooth(series):
            return series.rolling(7, center=True).mean()

        tracking_marker_smooth = signals_raw[tracking_marker].apply(smooth)

        # subtract smoothened from raw

        tracking_marker_diff = signals_raw[tracking_marker] - tracking_marker_smooth

        # collect peak times

        def get_peaks(timeseries, threshold):
            mean = timeseries.mean()
            std = timeseries.std()
            if threshold >= 0:
                peaks = timeseries.apply(lambda x: 1 if x > mean + (threshold * std) else 0)
            else:
                peaks = timeseries.apply(lambda x: 1 if x < mean + (threshold * std) else 0)
            return peaks

        tracking_marker_peaks = tracking_marker_diff.apply(get_peaks, args=(tracking_marker_peak_threshold,))
        #print("detected peaks")

        # Define cell divisions: Peak in iRFP signal AND drop subsequent in cell size
        cell_divisions = {}

        for cell in signals_raw[tracking_marker]:
            cell_divisions[cell] = []
            peak_times = list(tracking_marker_peaks[cell][tracking_marker_peaks[cell] == 1].index)
            peak_times_single = [time for time in peak_times if time - 1 not in peak_times]

            # print(peak_times_single)
            for peak_time in peak_times_single:
                # print(size_jumps[cell].loc[peak_time:peak_time+2])
                if -1 in list(size_jumps[cell].loc[peak_time:peak_time + int(round(2/(tracking_interval/60)))]):
                    cell_divisions[cell].append(peak_time)


        #

        def get_surrounding_timepoints(list_of_tps, rel_start, rel_end, interval):
            rel_start = int(round(rel_start / (interval / 60)))
            rel_end = int(round(rel_end / (interval / 60)))

            extended_list = []
            for tp in list_of_tps:
                extended_list += list(range(tp + rel_start, tp + rel_end + 1))
            return list(set([x for x in extended_list if x > 0]))

        # List size jumps that are not connected with divisions. Most cases either misslabeling or edge-effects
        non_division_size_jumps = {}
        #print("detected divisions")
        for cell in size_jumps:
            non_division_size_jumps[cell] = []
            size_jump_times = list(size_jumps[cell][size_jumps[cell] != 0].index)
            # print(size_jump_times)
            for size_jump_time in size_jump_times:

                if not size_jump_time in get_surrounding_timepoints(cell_divisions[cell], 0,
                                                                    2, tracking_interval):
                    non_division_size_jumps[cell].append(size_jump_time)


                # if not size_jump_time in (cell_divisions[cell] +
                #                           [x + 1 for x in cell_divisions[cell]] +
                #                           [x + 2 for x in cell_divisions[cell]]):
                #     non_division_size_jumps[cell].append(size_jump_time)

        # List tracking marker jumps not related to cell division
        non_division_irfp_jumps = {}
        for cell in tracking_marker_jumps:
            non_division_irfp_jumps[cell] = []
            irfp_jump_times = list(tracking_marker_jumps[cell][tracking_marker_jumps[cell] != 0].index)
            # print(size_jump_times)
            for irfp_jump_time in irfp_jump_times:
                if not irfp_jump_time in get_surrounding_timepoints(cell_divisions[cell],-2,
                                                                    2, tracking_interval):
                    non_division_irfp_jumps[cell].append(irfp_jump_time)
                # if not irfp_jump_time in (cell_divisions[cell] +
                #                           [x + 1 for x in cell_divisions[cell]] +
                #                           [x + 2 for x in cell_divisions[cell]] +
                #                           [x - 1 for x in cell_divisions[cell]] +
                #                           [x - 2 for x in cell_divisions[cell]]):
                #     non_division_irfp_jumps[cell].append(irfp_jump_time)

        # identify too close devisions, probably something went wrong
        def get_close_divisions(division_list, interval):
            close_divisions = []
            for cell, div_time_list in division_list.items():
                if len(div_time_list) > 1:
                    for position, div_time in enumerate(div_time_list[:-1]):
                        if (div_time_list[position + 1] - div_time_list[position]) < (15/(interval/60)):
                            close_divisions.append(cell)
                            break
            return close_divisions

        close_divisions = (get_close_divisions(cell_divisions, interval=tracking_interval))

        # print(rfp_signals.head())
        signals_cleaned = {}
        signals_cleaned_rel = {}

        # combine non_division_size_jumps and non_division_irfp_jumps
        # print(list(zip(non_division_size_jumps.keys(), non_division_irfp_jumps.keys())))

        flags_per_cell = {cell: list(set(non_division_size_jumps[cell]
                                         + non_division_irfp_jumps[cell]))
                          for cell in non_division_size_jumps.keys()}

        for color, raw_signals in signals_raw.items():
            signals_cleaned[color] = filter_cells(raw_signals,
                                                  flags_per_cell=flags_per_cell,
                                                  close_divisions=close_divisions,
                                                  min_len=min_len)
            signals_cleaned_rel[color] = signals_cleaned[color] / np.mean(signals_cleaned[color], axis=0)
        #print("cleaned all traces")
        all_cells = [cell for cell in signals_raw[tracking_marker]]
        approved_cells = [cell for cell in signals_cleaned[tracking_marker]]
        #print(f'{len(approved_cells) / len(all_cells) * 100} % of cells approved')

        # print("step0")
        # smoothen out cell division time points

        signals_smooth_clean_rel, signals_smooth_clean = {}, {}
        for color, raw_signals in signals_cleaned_rel.items():
            signals_smooth_clean_rel[color] = smoothen_out_divisions(signals_cleaned_rel[color], cell_divisions)
        for color, raw_signals in signals_cleaned.items():
            signals_smooth_clean[color] = smoothen_out_divisions(signals_cleaned[color], cell_divisions)

        # print("step1")
        nsubplots = 2 * len(main_channels) + 2

        fig, axs = plt.subplots(1, nsubplots, figsize=(30, 5))
        for i, color in enumerate(main_channels):

            #plot heatmap
            ax = axs[2 * i]
            df = signals_smooth_clean_rel[color].T

            df["tmax"] = df.iloc[:,:].idxmax(axis=1)
            sort = df.sort_values(by="tmax", axis=0).drop("tmax", axis=1)

            ax.imshow(sort, cmap="viridis", interpolation='nearest', vmin=(0.2), vmax=(2.6), aspect="auto")
            ax.set_title(F"{color} Rel. Signals")
            ax.set_ylabel("#cells")
            ax.set_xlabel("[h]")

            ax = axs[2 * i + 1]
            ax.plot(signals_smooth_clean_rel[color].mean(axis=1), c="red", label="mean")
            ax.plot(signals_smooth_clean_rel[color].median(axis=1), c="green", label="median")
            ax.legend()
            ax.set_title(F"{color} normalized")
            ax.set_ylabel("Amplitude")
            ax.set_xlabel("[h]")
            ax.set_ylim([0.3, 2.5])

        axs[nsubplots - 2].plot(signals_smooth_clean_rel[tracking_marker].count(axis=1))
        axs[nsubplots - 2].set_title("Cell Count")
        axs[nsubplots - 2].set_ylabel("# cells")
        axs[nsubplots - 2].set_xlabel("[h]")

        #print("generated plot")
        for i, color in enumerate(main_channels):
            axs[nsubplots - 1].boxplot(signals_cleaned[color].mean(), positions=[i + 1, ]),
        axs[nsubplots - 1].set_xticks(list(range(1, len(main_channels) + 1)))
        axs[nsubplots - 1].set_xticklabels(main_channels)
        axs[nsubplots - 1].set_title("Mean Signal")
        axs[nsubplots - 1].set_ylabel("a.u.")
        axs[nsubplots - 1].set_xlim(0.5, 2.5)

        for ax in fig.get_axes()[:-1]:
            make_ax_circadian(ax, signals_raw[tracking_marker], interval=tracking_interval)
        #plt.show()
        fig.suptitle(F'{dataset_name}')
        fig.savefig(output_folder + f'overview_accepted_cells_{dataset_name}.png', dpi=50)
        fig.savefig(output_folder + f'overview_accepted_cells_{dataset_name}(high_res).png', dpi=300)
        plt.close(fig)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(output_folder + f'post_script_output_{dataset_name}.xlsx')

        # Write each dataframe to a different worksheet.

        overview = [["cells", len(all_cells)],
                    ["approved_cells", len(approved_cells)],
                    ["minimum length", min_len]] + \
                   [[F'mean signal {color}', signals_cleaned[color].mean().mean()] for color in channels_to_color.values()]


        cell_divisions_flags = []
        for cell in signals_raw[tracking_marker]:
            cell_divisions_flags.append({"cell_number": cell,
                                         "divisions": cell_divisions[cell],
                                         "flags": list(set(non_division_size_jumps[cell] +
                                                           non_division_irfp_jumps[cell])),
                                         "approved": cell in approved_cells,
                                         })

        pd.DataFrame(overview).to_excel(writer, sheet_name='overview')
        pd.DataFrame(cell_divisions_flags).to_excel(writer, sheet_name='div_flags')
        for color in channels_to_color.values():
            signals_raw[color].to_excel(writer, sheet_name=F'{color}_raw_all_cells')
            signals_cleaned[color].to_excel(writer, sheet_name=F'{color}_raw_acpt_cells')
            signals_smooth_clean[color].to_excel(writer, sheet_name=F'{color}_raw_acpt_cells_smoothDiv')
            signals_smooth_clean_rel[color].to_excel(writer, sheet_name=F'{color}_norm_acpt_cells_smoothDiv')

        writer.close()

    except Exception:
        results["error"] = traceback.format_exc()
        return results

    results["output_folder"]=output_folder
    results["run_complete"]=True
    results["all_cells"]=len(all_cells)
    results["approved_cells"]= len(approved_cells)
    results["fig_path"]= output_folder + f'overview_accepted_cells_{dataset_name}.png'

    return results

def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def filter_cells(dataframe, flags_per_cell, close_divisions, min_len):

    filtered_df = dataframe.copy(deep=True)
    df_length = dataframe.shape[0]
    for cell, flag_list in flags_per_cell.items():
        #print(cell)
        cell_time_series = dataframe[cell].dropna()
        for flag in flag_list:
            #print(cell,flag)
            if cell_time_series.loc[flag+1:].shape > cell_time_series.loc[:flag-1].shape:
                cell_time_series = cell_time_series.loc[flag+1:]
            else:
                cell_time_series = cell_time_series.loc[:flag-1]
        if cell_time_series.shape[0] < min_len:

            #print("removed")
            #print(filtered_df.columns)
            filtered_df.drop(cell, axis =1, inplace=True)
        elif cell in close_divisions:
            #print(filtered_df.columns)
            filtered_df.drop(cell, axis =1, inplace=True)
        else:
            #print("kept")
            filtered_df[cell] = cell_time_series
    return filtered_df

# smoothen out cell division time points
def smoothen_out_divisions(dataset, cell_divisions):
    df = dataset.copy(deep=True)
    for cell, division_list in cell_divisions.items():
        try:
            for division_time in division_list:

                div_plus1 = division_time +1
                new_y_diff_time = (df[cell].loc[division_time-1] +
                                   (df[cell].loc[division_time+2]- df[cell].loc[division_time-1])*0.33)

                new_y_diff_time_plus_1 = (df[cell].loc[division_time-1] +
                                   (df[cell].loc[division_time+2]- df[cell].loc[division_time-1])*0.66)

                df.loc[division_time, cell] = new_y_diff_time
                df.loc[division_time+1, cell] = new_y_diff_time_plus_1

        except:
            continue
    return df

def make_ax_circadian(ax, irfp_signals, interval):
    ax.set_xlabel("[h]")
    data_count = irfp_signals.shape[0]
    max_time = int(data_count*interval/60)
    x_ticks = np.array(range(0,max_time,12), dtype=float)/(interval/60)
    x_tick_labels = list(range(0,max_time,12))

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlim(0,data_count)

    ax.grid()


def difference_to_prev(time_series):
    return time_series.diff()/time_series


def max_min_null(value, max_cutoff, min_cutoff=None):
    if not min_cutoff:
        min_cutoff= -1 * max_cutoff
    if value >= max_cutoff:
        return 1
    elif value <=min_cutoff:
        return -1
    else:
        return 0

def get_jumps(time_series, cutoff):
    return time_series.apply(max_min_null, args =(cutoff,))

def erase_number(name):
    pattern = re.compile('[0-9]+')
    if re.match(pattern, name.split("_")[-1]):
        name = "_".join(name.split("_")[:-1])
    return name

def add_dataset_number(name, dataset_nr):
    return f'{dataset_nr}_{name}'



def df_to_numeric(df):
    for x in df.columns:
        df[x]=pd.to_numeric(df[x])
    return df

def extract_raw_time_series(cell_df, channels_to_color, min_len):
    signals_raw = {}
    colors_raw_extracted = []
    for channel, color in channels_to_color.items():
        signals_raw[color] = (cell_df.drop_duplicates(subset=["TRACK_ID", "FRAME"])
                              .pivot(index="FRAME",
                                                columns="TRACK_ID",
                                                    values=f'MEAN_INTENSITY_CH{channel}' )
                              .reset_index().reset_index(drop=True))

        signals_raw[color].FRAME = pd.to_numeric(signals_raw[color].FRAME)
        signals_raw[color]= (df_to_numeric(signals_raw[color].sort_values(by="FRAME"))
                             .set_index("FRAME", drop=True))
        #drop time series shorter than limit
        signals_raw[color]= signals_raw[color].loc[:, signals_raw[color].count(axis=0)>min_len]
        colors_raw_extracted.append(color)

    #print(f'  extracted BG subtracted data from channels: {colors_raw_extracted}')

    return signals_raw


def extract_size_time_series(cell_df, min_len):
    sizes = cell_df.drop_duplicates(subset=["TRACK_ID", "FRAME"]).pivot(index="FRAME",
                                                columns="TRACK_ID",
                                                    values=f'AREA' ).reset_index().reset_index(drop=True)

    sizes.FRAME = pd.to_numeric(sizes.FRAME)
    sizes =sizes.sort_values(by="FRAME").set_index("FRAME")
    #print(sizes)
    #print(sizes.count(axis=0))
    sizes = sizes.loc[:, sizes.count(axis=0)>min_len]
    return df_to_numeric(sizes)