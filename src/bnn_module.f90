module bnn_module
    use hdf5
    implicit none
    private
    public :: initialize_network, load_weights, load_metadata, predict, standardize, unstandardize, set_random_seed
    public :: relu, tanh_fn, elu, no_activation, layer_activations

    ! Define abstract interface for activation functions
    abstract interface
        function activation_func_interface(x)

            real, dimension(:), intent(in) :: x
            real, dimension(size(x)) :: activation_func_interface
        end function activation_func_interface
    end interface

    ! A type to hold a procedure pointer to an activation function
    type activation_holder
        procedure(activation_func_interface), pointer, nopass :: func
    end type activation_holder

    ! Define a layer type to hold parameters for each layer
    type layer
        real, allocatable :: kernel_posterior_loc(:,:)
        real, allocatable :: kernel_posterior_scale(:,:)
        real, allocatable :: bias_posterior_loc(:)
        real, allocatable :: bias_posterior_scale(:)
    end type layer

    ! Dynamically sized arrays defining the network structure and activation functions
    integer, allocatable :: layer_sizes(:)
    type(layer), allocatable :: network(:)

    ! Array of activation holders for each layer
    type(activation_holder), allocatable :: layer_activations(:)

    ! Create a generic interface for sample_normal
    interface sample_normal
        module procedure sample_normal_2d, sample_normal_1d
    end interface sample_normal

contains

    !------------------------------------------------------------
    ! Initialization routine to define the network architecture
    !------------------------------------------------------------
    subroutine initialize_network(sizes)
        integer, intent(in) :: sizes(:)
        integer :: i, network_depth

        if (allocated(layer_sizes)) deallocate(layer_sizes)
        layer_sizes = sizes
        network_depth = size(layer_sizes)-1

        if (allocated(network)) deallocate(network)
        allocate(network(network_depth))

        ! Allocate arrays for each layer based on layer_sizes
        do i = 1, network_depth
            allocate(network(i)%kernel_posterior_loc(layer_sizes(i+1), layer_sizes(i)))
            allocate(network(i)%kernel_posterior_scale(layer_sizes(i+1), layer_sizes(i)))
            allocate(network(i)%bias_posterior_loc(layer_sizes(i+1)))
            allocate(network(i)%bias_posterior_scale(layer_sizes(i+1)))
        end do

        ! Allocate and assign default activations
        if (allocated(layer_activations)) deallocate(layer_activations)
        allocate(layer_activations(network_depth))

        ! Default: relu for all but last layer, last layer no activation
        do i = 1, network_depth-1
            layer_activations(i)%func => relu
        end do
        layer_activations(network_depth)%func => no_activation
    end subroutine initialize_network

    !------------------------------------------------------------
    ! Load the network weights from an HDF5 file
    !------------------------------------------------------------
    subroutine load_weights(model_path)
        character(len=*), intent(in) :: model_path
        integer(hid_t) :: file_id
        integer :: i, hdferr
        character(256) :: dataset_name
        logical :: exists
        integer :: network_depth

        network_depth = size(layer_sizes)-1

        call h5open_f(hdferr)
        call h5fopen_f(trim(model_path), H5F_ACC_RDONLY_F, file_id, hdferr)

        do i = 1, network_depth
            ! Kernel loc
            dataset_name = get_dataset_name(i, "kernel_posterior_loc:0")
            call load_dataset(file_id, dataset_name, network(i)%kernel_posterior_loc)

            ! Kernel scale
            dataset_name = get_dataset_name(i, "kernel_posterior_untransformed_scale:0")
            call load_dataset(file_id, dataset_name, network(i)%kernel_posterior_scale)

            ! Bias loc
            dataset_name = get_dataset_name(i, "bias_posterior_loc:0")
            call load_dataset_1d(file_id, dataset_name, network(i)%bias_posterior_loc)

            ! Bias scale
            dataset_name = get_dataset_name(i, "bias_posterior_untransformed_scale:0")
            call dataset_exists(file_id, dataset_name, exists)
            if (exists) then
                call load_dataset_1d(file_id, dataset_name, network(i)%bias_posterior_scale)
            else
                network(i)%bias_posterior_scale = 0.0
                print *, "Warning: Dataset not found for layer", i, "bias scale. Defaulting to zero."
            end if
        end do

        call h5fclose_f(file_id, hdferr)
        call h5close_f(hdferr)
    end subroutine load_weights

    !------------------------------------------------------------
    ! Load metadata for standardization
    !------------------------------------------------------------
    subroutine load_metadata(metadata_path, x_mean, y_mean, x_std, y_std)
        character(len=*), intent(in) :: metadata_path
        real, allocatable, intent(out) :: x_mean(:), y_mean(:), x_std(:), y_std(:)
        integer(hid_t) :: file_id
        integer :: hdferr

        call h5open_f(hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening HDF5 library: ', hdferr
            stop 'Error opening HDF5 library.'
        end if

        call h5fopen_f(metadata_path, H5F_ACC_RDONLY_F, file_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening HDF5 file:', metadata_path, 'Error code:', hdferr
            stop 'Error opening HDF5 file.'
        end if

        ! Read scaler information
        call load_dataset_1d(file_id, 'scaler/x_mean', x_mean)
        call load_dataset_1d(file_id, 'scaler/y_mean', y_mean)
        call load_dataset_1d(file_id, 'scaler/x_std', x_std)
        call load_dataset_1d(file_id, 'scaler/y_std', y_std)

        call h5fclose_f(file_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing HDF5 file:', metadata_path, 'Error code:', hdferr
            stop 'Error closing HDF5 file.'
        end if

        call h5close_f(hdferr)
        if (hdferr /= 0) then
            print *, 'Error closing HDF5 library: ', hdferr
            stop 'Error closing HDF5 library.'
        end if
    end subroutine load_metadata

    !------------------------------------------------------------
    ! Helper to generate dataset names based on layer index
    !------------------------------------------------------------
    function get_dataset_name(layer_index, param_name) result(name)
        integer, intent(in) :: layer_index
        character(len=*), intent(in) :: param_name
        character(len=256) :: name

        if (layer_index == 1) then
            write(name, '(A)') 'model_weights/dense_flipout/dense_flipout/'//trim(param_name)
        else
            write(name, '(A,"/dense_flipout_",I0,"/dense_flipout_",I0,"/",A)') 'model_weights', layer_index-1, layer_index-1, param_name
        end if
    end function get_dataset_name

    !------------------------------------------------------------
    ! Load a 2D dataset from the HDF5 file
    !------------------------------------------------------------
    subroutine load_dataset(file_id, dataset_name, data)
        integer(hid_t), intent(in) :: file_id
        character(len=*), intent(in) :: dataset_name
        real, allocatable, intent(out) :: data(:,:)

        integer(hid_t) :: dataset_id, dataspace_id
        integer :: hdferr
        integer(HSIZE_T) :: dims(2), maxdims(2)

        call h5dopen_f(file_id, dataset_name, dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening dataset:', dataset_name, 'Code:', hdferr
            stop 'Error opening dataset.'
        end if

        call h5dget_space_f(dataset_id, dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error getting dataspace for:', dataset_name, 'Code:', hdferr
            stop 'Error getting dataspace.'
        end if

        call h5sget_simple_extent_dims_f(dataspace_id, dims, maxdims, hdferr)
        if (hdferr == -1) then
            print *, 'Error getting dimensions for:', dataset_name
            stop 'Error getting dimensions.'
        end if

        allocate(data(dims(1), dims(2)))
        call h5dread_f(dataset_id, H5T_NATIVE_REAL, data, dims, hdferr)
        if (hdferr /= 0) then
            print *, 'Error reading dataset:', dataset_name, 'Code:', hdferr
            stop 'Error reading dataset.'
        end if

        call h5sclose_f(dataspace_id, hdferr)
        call h5dclose_f(dataset_id, hdferr)
    end subroutine load_dataset

    !------------------------------------------------------------
    ! Load a 1D dataset from the HDF5 file
    !------------------------------------------------------------
    subroutine load_dataset_1d(file_id, dataset_name, data)
        integer(hid_t), intent(in) :: file_id
        character(len=*), intent(in) :: dataset_name
        real, allocatable, intent(out) :: data(:)

        integer(hid_t) :: dataset_id, dataspace_id
        integer :: hdferr
        integer(HSIZE_T) :: dims(1), maxdims(1)

        call h5dopen_f(file_id, dataset_name, dataset_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error opening dataset:', dataset_name
            stop 'Error opening dataset.'
        end if

        call h5dget_space_f(dataset_id, dataspace_id, hdferr)
        if (hdferr /= 0) then
            print *, 'Error getting dataspace:', dataset_name
            stop 'Error getting dataspace.'
        end if

        call h5sget_simple_extent_dims_f(dataspace_id, dims, maxdims, hdferr)
        if (hdferr == -1) then
            print *, 'Error getting dimensions for:', dataset_name
            stop 'Error getting dimensions.'
        end if

        allocate(data(dims(1)))
        call h5dread_f(dataset_id, H5T_NATIVE_REAL, data, dims, hdferr)
        if (hdferr /= 0) then
            print *, 'Error reading dataset:', dataset_name
            stop 'Error reading dataset.'
        end if

        call h5sclose_f(dataspace_id, hdferr)
        call h5dclose_f(dataset_id, hdferr)
    end subroutine load_dataset_1d

    !------------------------------------------------------------
    ! Check if dataset exists in the file
    !------------------------------------------------------------
    subroutine dataset_exists(file_id, dataset_name, exists)
        integer(hid_t), intent(in) :: file_id
        character(len=*), intent(in) :: dataset_name
        logical, intent(out) :: exists

        integer :: hdferr
        logical :: link_exists

        call H5Lexists_f(file_id, trim(dataset_name), link_exists, hdferr)
        if (hdferr /= 0) then
            print *, "Error checking existence of:", trim(dataset_name)
            stop "Error in H5Lexists_f"
        end if

        exists = link_exists
    end subroutine dataset_exists

    !------------------------------------------------------------
    ! Set the random seed (useful for reproducibility)
    !------------------------------------------------------------
    subroutine set_random_seed(seed)
        integer, intent(in) :: seed
        integer, allocatable :: seed_array(:)
        integer :: i, n

        call random_seed(size=n)
        allocate(seed_array(n))
        do i = 1, n
            seed_array(i) = mod(seed + i - 1, 2147483647)
        end do
        call random_seed(put=seed_array)
    end subroutine set_random_seed

    !------------------------------------------------------------
    ! Standardization and unstandardization routines
    !------------------------------------------------------------
    subroutine standardize(data, mean, std)
        real, intent(inout) :: data(:)
        real, intent(in) :: mean, std
        integer :: i
        do i = 1, size(data)
            data(i) = (data(i) - mean) / std
        end do
    end subroutine standardize

    subroutine unstandardize(data, mean, std)
        real, intent(inout) :: data(:)
        real, intent(in) :: mean, std
        integer :: i
        do i = 1, size(data)
            data(i) = data(i)*std + mean
        end do
    end subroutine unstandardize

    !------------------------------------------------------------
    ! Sampling from a normal distribution
    !------------------------------------------------------------
    function sample_normal_2d(mean, scale) result(sample)
        real, intent(in) :: mean(:,:), scale(:,:)
        real, allocatable :: sample(:,:)
        real, allocatable :: epsilon(:,:)
        integer :: nrows, ncols

        nrows = size(mean,1)
        ncols = size(mean,2)
        allocate(epsilon(nrows,ncols), sample(nrows,ncols))

        call random_number(epsilon)
        sample = mean + log(1.0 + exp(scale)) * (epsilon - 0.5) * 2.0
    end function sample_normal_2d

    function sample_normal_1d(mean, scale) result(sample)
        real, intent(in) :: mean(:), scale(:)
        real :: sample(size(mean))
        real :: epsilon(size(mean))

        call random_number(epsilon)
        sample = mean + log(1.0 + exp(scale)) * (epsilon - 0.5) * 2.0
    end function sample_normal_1d

    !------------------------------------------------------------
    ! Activation functions
    !------------------------------------------------------------
    function relu(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = max(0.0, x)
    end function relu

    function tanh_fn(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = tanh(x)
    end function tanh_fn

    function elu(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = x
        where (x < 0.0) y = exp(x) - 1.0
    end function elu

    function no_activation(x) result(y)
        real, dimension(:), intent(in) :: x
        real, dimension(size(x)) :: y
        y = x
    end function no_activation

    !------------------------------------------------------------
    ! Predict using the network:
    ! Each layer uses its own activation function from layer_activations.
    !------------------------------------------------------------
    function predict(input) result(output)
        real, dimension(:), intent(in) :: input
        real, allocatable :: output(:)
        real, allocatable :: layer_output(:)
        real, dimension(:), allocatable :: current(:)
        integer :: i, network_depth

        network_depth = size(layer_sizes)-1
        allocate(current(size(input)))
        current = input

        do i = 1, network_depth
            ! Compute linear transformation: W * current + b
            layer_output = matmul(sample_normal(network(i)%kernel_posterior_loc, &
                                                network(i)%kernel_posterior_scale), current) + &
                           sample_normal(network(i)%bias_posterior_loc, &
                                         network(i)%bias_posterior_scale)

            ! Apply layer-specific activation
            layer_output = layer_activations(i)%func(layer_output)

            if (allocated(current)) deallocate(current)
            allocate(current(size(layer_output)))
            current = layer_output
            deallocate(layer_output)
        end do

        allocate(output(size(current)))
        output = current
        deallocate(current)
    end function predict

end module bnn_module