/**
 * @fileOverview Classes for each selector of the main page.
 * @author Antonio Strippoli
 */


/**
 * Builds a BootstrapSelect object which will be used as Model selector.
 * @param {Array.<string>} elements The options.
 */
export class ModelSelector {
    constructor(elements) {
        this.selector = $('.selectpicker')

        var options_html = ''
        for (var i in elements) {
            options_html += "<option>" + elements[i] + "</option>"
        }

        this.selector.html(options_html).selectpicker('refresh')
    }
}


/**
 * Builds a noUISlider which will be used as Training Step selector.
 * @param {Array.<string>} elements The pips' label.
 */
export class TrainingStepSelector {
    constructor(elements) {
        this.slider = document.getElementsByClassName('slider')[0]

        noUiSlider.create(this.slider, {
            start: 0,
            connect: [true, false],
            range: {
                min: 0,
                max: elements.length - 1
            },
            step: 1,
            pips: {
                mode: 'steps',
                stepped: true,
                density: elements.length,
                format: {
                    to: function (value) {
                        return elements[Math.round(value)]
                    },
                    from: Number
                }
            },
        })
    }

    /**
     * Enable the slider.
     */
    enable() { this.slider.removeAttribute('disabled') }

    /**
     * Disable the slider.
     */
    disable() { this.slider.setAttribute('disabled', true) }

    /**
     * Change the elements of a slider.
     * @param {Array.<string>} elements The new elements for the slider.
     */
    changeElements(elements) {
        this.slider.noUiSlider.updateOptions({
            range: {
                min: 0,
                max: elements.length - 1
            },
            pips: {
                mode: 'steps',
                stepped: true,
                density: elements.length,
                format: {
                    to: function (value) {
                        return elements[Math.round(value)]
                    },
                    from: Number
                }
            },
        })
    }
}
