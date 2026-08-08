"""Tests for quick_text, the automatic label formatter."""

import pytest

from PythonAQ import quick_text


class TestSpecies:
    @pytest.mark.parametrize('raw,expected', [
        ('no2', 'NO<sub>2</sub>'),
        ('NO2', 'NO<sub>2</sub>'),
        ('No2', 'NO<sub>2</sub>'),
        ('nox', 'NO<sub>x</sub>'),
        ('o3', 'O<sub>3</sub>'),
        ('so2', 'SO<sub>2</sub>'),
        ('pm10', 'PM<sub>10</sub>'),
        ('pm2.5', 'PM<sub>2.5</sub>'),
        ('pm25', 'PM<sub>2.5</sub>'),
        ('co', 'CO'),
        ('co2', 'CO<sub>2</sub>'),
    ])
    def test_known_species(self, raw, expected):
        assert quick_text(raw) == expected

    def test_longest_match_wins(self):
        """'NOXasNO2' must not be chopped up by the shorter 'no' or 'nox'."""
        assert quick_text('NOXasNO2') == 'NO<sub>x</sub> as NO<sub>2</sub>'

    def test_meteorological_columns(self):
        assert quick_text('ws') == 'wind speed'
        assert quick_text('wd') == 'wind direction'
        assert quick_text('temp') == 'temperature'


class TestUnits:
    @pytest.mark.parametrize('raw,expected', [
        ('ug/m3', 'µg m<sup>-3</sup>'),
        ('mg/m3', 'mg m<sup>-3</sup>'),
        ('m/s', 'm s<sup>-1</sup>'),
    ])
    def test_units(self, raw, expected):
        assert quick_text(raw) == expected

    def test_species_and_units_together(self):
        assert quick_text('PM2.5 (ug/m3)') == 'PM<sub>2.5</sub> (µg m<sup>-3</sup>)'


class TestSafety:
    @pytest.mark.parametrize('text', [
        'nothing',        # contains 'no'
        'Nottingham',     # contains 'no'
        'carbon',         # contains 'co'
        'concentration',  # contains 'co' and 'o3'-adjacent characters
        'cost',           # contains 'co'
        'Site 3 kerbside',
        'Leeds Centre',
    ])
    def test_does_not_match_inside_words(self, text):
        """Species codes are short; matching them mid-word would mangle prose."""
        assert quick_text(text) == text

    def test_substitution_happens_only_on_whole_tokens(self):
        assert quick_text('mean NO2 by hour') == 'mean NO<sub>2</sub> by hour'

    @pytest.mark.parametrize('value', [None, 42, 3.5, ['no2']])
    def test_non_strings_pass_through(self, value):
        assert quick_text(value) is value or quick_text(value) == value

    def test_empty_string(self):
        assert quick_text('') == ''

    def test_auto_false_disables_formatting(self):
        assert quick_text('no2', auto=False) == 'no2'

    def test_plain_strips_markup(self):
        assert quick_text('pm2.5', plain=True) == 'PM2.5'
        assert '<' not in quick_text('PM2.5 (ug/m3)', plain=True)

    def test_is_idempotent_on_already_formatted_text(self):
        """Running it twice must not nest tags."""
        once = quick_text('no2')
        assert quick_text(once) == once


class TestUsedByPlots:
    def test_plot_labels_are_formatted(self, aq_df):
        from PythonAQ import percentile_rose, polar_plot, scatter_plot

        fig = polar_plot(aq_df, conc_col='NO2')
        assert fig.data[0].colorbar.title.text == 'NO<sub>2</sub>'

        fig, _ = percentile_rose(aq_df, 'NO2')
        assert 'NO<sub>2</sub>' in fig.layout.title.text

        fig, _ = scatter_plot(aq_df, 'NO2', 'PM10')
        assert fig.layout.xaxis.title.text == 'NO<sub>2</sub>'
        assert fig.layout.yaxis.title.text == 'PM<sub>10</sub>'
